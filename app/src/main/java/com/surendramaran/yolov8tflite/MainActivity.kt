package com.surendramaran.yolov8tflite

import android.Manifest
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.surendramaran.yolov8tflite.Constants.LABELS_PATH
import com.surendramaran.yolov8tflite.Constants.MODEL_PATH
import com.surendramaran.yolov8tflite.databinding.ActivityMainBinding
import kotlinx.coroutines.launch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), Detector.DetectorListener {
    private lateinit var binding: ActivityMainBinding

    // Camera & Detection
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var detector: Detector
    private lateinit var cameraExecutor: ExecutorService
    private var cameraFacing = CameraSelector.LENS_FACING_BACK

    // Performance Buffer
    private var bitmapBuffer: Bitmap? = null

    // AI & Logic
    private lateinit var coachBrain: CoachBrain
    private var isRecording = false
    private val sessionData = mutableListOf<FrameData>()
    private var sessionStartTime = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // 1. Setup Detector (GPU=true, NNAPI=false)
        detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this, true, false)
        detector.setup()

        // 2. Setup AI Brain
        coachBrain = CoachBrain(this)
        lifecycleScope.launch {
            coachBrain.initialize()
        }

        // 3. Setup UI Listeners
        setupUI()

        // 4. Start Camera
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun setupUI() {
        // Mode Selection Chips
        binding.chipSquat.setOnClickListener {
            // Future: Set currentMode = "SQUAT"
            // binding.inferenceTime.text = "Mode: Squat"
        }

        // History Button
        binding.btnHistory.setOnClickListener {
            val intent = Intent(this, HistoryActivity::class.java)
            startActivity(intent)
        }

        // Camera Switch
        binding.btnSwitchCamera.setOnClickListener {
            cameraFacing = if (cameraFacing == CameraSelector.LENS_FACING_BACK) {
                CameraSelector.LENS_FACING_FRONT
            } else {
                CameraSelector.LENS_FACING_BACK
            }
            startCamera()
        }

        // Record / Stop Button
        binding.recordButton.setOnClickListener {
            if (!isRecording) {
                startRecording()
            } else {
                stopRecordingAndAnalyze()
            }
        }
    }

    private fun startRecording() {
        isRecording = true
        sessionData.clear()
        sessionStartTime = SystemClock.elapsedRealtime()

        // UI Updates: Red Pulse Icon
        binding.recordButton.setImageResource(android.R.drawable.ic_media_pause) // Change to a "Stop" icon
        binding.recordButton.imageTintList = ColorStateList.valueOf(Color.RED)
        binding.timerText.visibility = View.VISIBLE
        binding.timerText.text = "00:00"
    }

    private fun stopRecordingAndAnalyze() {
        isRecording = false
        val durationSec = (SystemClock.elapsedRealtime() - sessionStartTime) / 1000

        // UI Reset
        binding.recordButton.setImageResource(R.drawable.ic_circle) // Back to circle
        binding.recordButton.imageTintList = ColorStateList.valueOf(Color.WHITE)
        binding.timerText.visibility = View.GONE

        // 1. Show "Thinking" Dialog
        val loadingDialog = AlertDialog.Builder(this)
            .setTitle("Generating Biomechanical Report")
            .setMessage("The AI Coach is analyzing your movement vectors...")
            .setCancelable(false)
            .show()

        // 2. Run Pipeline (Background)
        lifecycleScope.launch {
            // Step A: Math (Instant)
            val mathSummary = PoseUtils.analyzeSession(sessionData)

            // Step B: Brain (LLM Generation)
            val aiText = coachBrain.generateAnalysis(
                reps = mathSummary.totalReps,
                score = mathSummary.accuracy.toInt(),
                exercise = "Squat"
            )

            // Step C: Cleanup & Show Result
            loadingDialog.dismiss()
            val cleanText = aiText.replace("**", "").replace("##", "").trim()

            showSmartSummary(mathSummary, durationSec, cleanText)
        }
    }

    private fun showSmartSummary(summary: WorkoutSummary, duration: Long, aiText: String) {
        AlertDialog.Builder(this)
            .setTitle("Workout Complete")
            .setMessage("""
                Reps: ${summary.totalReps}
                Form Score: ${summary.accuracy.toInt()}%
                Duration: ${duration}s
                
                $aiText
            """.trimIndent())
            .setPositiveButton("Save to History") { _, _ ->
                // Save to Room DB in Background Thread
                Thread {
                    val db = AppDatabase.getDatabase(this)
                    val session = WorkoutSession(
                        timestamp = System.currentTimeMillis(),
                        exerciseType = "Squat",
                        reps = summary.totalReps,
                        formScore = summary.accuracy.toInt(),
                        durationSec = duration,
                        feedback = aiText
                    )
                    db.workoutDao().insert(session)
                }.start()
            }
            .setNegativeButton("Discard", null)
            .show()
    }

    // --- DETECTOR CALLBACK (The Loop) ---
    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long, device: String) {
        runOnUiThread {
            binding.overlay.setResults(boundingBoxes)

            if (boundingBoxes.isNotEmpty()) {
                val person = boundingBoxes[0]
                val k = person.keypoints

                // Check visibility of Right Leg (Hip=12, Knee=14, Ankle=16)
                if (k[12].confidence > 0.5 && k[14].confidence > 0.5 && k[16].confidence > 0.5) {
                    val angle = PoseUtils.calculateAngle(k[12], k[14], k[16])

                    // Live Feedback UI
                    if (angle < 100) {
                        binding.inferenceTime.text = "Perfect Depth! (${angle.toInt()}°)"
                        binding.statsCard.setCardBackgroundColor(0xFF4CAF50.toInt()) // Green
                    } else if (angle < 140) {
                        binding.inferenceTime.text = "Go Lower... (${angle.toInt()}°)"
                        binding.statsCard.setCardBackgroundColor(0xFFFFC107.toInt()) // Yellow
                    } else {
                        binding.inferenceTime.text = "Stand Tall (${angle.toInt()}°)"
                        binding.statsCard.setCardBackgroundColor(0x80000000.toInt()) // Transparent
                    }

                    // Recording Logic
                    if (isRecording) {
                        sessionData.add(FrameData(System.currentTimeMillis(), person.keypoints, angle))

                        // Update Timer
                        val duration = (SystemClock.elapsedRealtime() - sessionStartTime) / 1000
                        binding.timerText.text = "${duration / 60}:${String.format("%02d", duration % 60)}"
                    }
                } else {
                    binding.inferenceTime.text = "Adjust Camera"
                    binding.statsCard.setCardBackgroundColor(0x80000000.toInt())
                }
            } else {
                binding.inferenceTime.text = "Searching..."
                binding.statsCard.setCardBackgroundColor(0x80000000.toInt())
            }
        }
    }

    override fun onEmptyDetect() {
        runOnUiThread {
            binding.overlay.invalidate()
            binding.inferenceTime.text = "Searching..."
        }
    }

    // --- CAMERA BOILERPLATE ---
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: return
        val rotation = binding.viewFinder.display.rotation
        val cameraSelector = CameraSelector.Builder().requireLensFacing(cameraFacing).build()

        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
            // Buffer Optimization
            if (bitmapBuffer == null || bitmapBuffer!!.width != imageProxy.width || bitmapBuffer!!.height != imageProxy.height) {
                bitmapBuffer = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
            }

            val buffer = imageProxy.planes[0].buffer
            buffer.rewind()
            bitmapBuffer!!.copyPixelsFromBuffer(buffer)
            imageProxy.close()

            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                if (cameraFacing == CameraSelector.LENS_FACING_FRONT) {
                    postScale(-1f, 1f, imageProxy.width.toFloat(), imageProxy.height.toFloat())
                }
            }

            val rotatedBitmap = Bitmap.createBitmap(bitmapBuffer!!, 0, 0, bitmapBuffer!!.width, bitmapBuffer!!.height, matrix, true)
            detector.detect(rotatedBitmap)
        }

        cameraProvider.unbindAll()
        camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
        preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        if (it[Manifest.permission.CAMERA] == true) startCamera()
    }

    override fun onDestroy() {
        super.onDestroy()
        detector.clear()
        cameraExecutor.shutdown()
    }

    override fun onResume() {
        super.onResume()
        if (allPermissionsGranted()) startCamera()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = mutableListOf(Manifest.permission.CAMERA).toTypedArray()
    }
}
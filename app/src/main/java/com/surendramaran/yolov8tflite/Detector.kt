package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader

class Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String,
    private val detectorListener: DetectorListener,
    private val isGPU: Boolean = false,
    private val isNNAPI: Boolean = false
) {
    private var interpreter: Interpreter? = null
    private var labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    fun setup() {
        val model = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options()

        if (isGPU) {
            try {
                options.addDelegate(GpuDelegate())
            } catch (e: Exception) {
                e.printStackTrace()
            }
        } else {
            options.numThreads = 4
        }

        if (isNNAPI) {
            options.setUseNNAPI(true)
        }

        interpreter = Interpreter(model, options)

        val inputShape = interpreter?.getInputTensor(0)?.shape() ?: return
        val outputShape = interpreter?.getOutputTensor(0)?.shape() ?: return

        tensorWidth = inputShape[1]
        tensorHeight = inputShape[2]
        numChannel = outputShape[1]
        numElements = outputShape[2]

        try {
            val inputStream: InputStream = context.assets.open(labelPath)
            val reader = BufferedReader(InputStreamReader(inputStream))
            var line: String? = reader.readLine()
            while (line != null && line != "") {
                labels.add(line)
                line = reader.readLine()
            }
            reader.close()
            inputStream.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    fun clear() {
        interpreter?.close()
        interpreter = null
    }

    fun detect(frame: Bitmap) {
        interpreter ?: return
        if (tensorWidth == 0) return

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer
        val output = TensorBuffer.createFixedSize(intArrayOf(1 , numChannel, numElements), OUTPUT_IMAGE_TYPE)

        val startTime = SystemClock.uptimeMillis()
        interpreter?.run(imageBuffer, output.buffer)
        val gpuFinishTime = SystemClock.uptimeMillis()

        val bestBoxes = bestBox(output.floatArray)
        val processFinishTime = SystemClock.uptimeMillis()

        val gpuTime = gpuFinishTime - startTime
        val cpuTime = processFinishTime - gpuFinishTime
        val deviceStr = if (isGPU) "GPU" else if (isNNAPI) "NNAPI" else "CPU"
        val debugText = "$deviceStr (Inf: ${gpuTime}ms | Post: ${cpuTime}ms)"

        if (bestBoxes == null) {
            detectorListener.onEmptyDetect()
            return
        }

        detectorListener.onDetect(bestBoxes, gpuTime + cpuTime, debugText)
    }

    private fun bestBox(array: FloatArray) : List<BoundingBox>? {
        val boundingBoxes = mutableListOf<BoundingBox>()

        for (c in 0 until numElements) {
            val conf = array[c + numElements * 4]
            if (conf > CONFIDENCE_THRESHOLD) {
                val cx = array[c]
                val cy = array[c + numElements]
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]
                val x1 = cx - (w/2F)
                val y1 = cy - (h/2F)
                val x2 = cx + (w/2F)
                val y2 = cy + (h/2F)
                if (x1 < 0F || x1 > 1F || y1 < 0F || y1 > 1F || x2 < 0F || x2 > 1F || y2 < 0F || y2 > 1F) continue

                val keypoints = mutableListOf<Keypoint>()
                for (k in 0 until 17) {
                    val kIdx = 5 + (k * 3)
                    val kx = array[c + numElements * kIdx]
                    val ky = array[c + numElements * (kIdx + 1)]
                    val kConf = array[c + numElements * (kIdx + 2)]
                    keypoints.add(Keypoint(kx, ky, kConf))
                }

                // --- LOGIC TEST: SQUAT CHECK ---
                // Hip=11, Knee=13, Ankle=15 (Left Side)
                if (keypoints.size > 15) {
                    val hip = keypoints[11]
                    val knee = keypoints[13]
                    val ankle = keypoints[15]

                    if (hip.confidence > 0.5 && knee.confidence > 0.5 && ankle.confidence > 0.5) {
                        val kneeAngle = PoseUtils.calculateAngle(hip, knee, ankle)

                        // Simple Logic: If angle < 90, it's a deep squat
                        val status = if (kneeAngle < 100) "DEEP SQUAT!" else "Standing"

                        // Print to Logcat for now (Filter by "SQUAT")
                        android.util.Log.d("SQUAT", "Angle: ${kneeAngle.toInt()}° -> $status")
                    }
                }

                boundingBoxes.add(
                    BoundingBox(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cx = cx, cy = cy, w = w, h = h,
                        cnf = conf, cls = 0, clsName = "Person",
                        keypoints = keypoints
                    )
                )
            }
        }
        if (boundingBoxes.isEmpty()) return null
        return applyNMS(boundingBoxes)
    }

    private fun applyNMS(boxes: List<BoundingBox>) : MutableList<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()
        while(sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)
            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou >= IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }
        return selectedBoxes
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    interface DetectorListener {
        fun onEmptyDetect()
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long, device: String)
    }

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.5F
        private const val IOU_THRESHOLD = 0.5F

        // ↓↓↓ PASTE THIS HERE ↓↓↓
        fun calculateAngle(a: Keypoint, b: Keypoint, c: Keypoint): Double {
            val radians = Math.atan2((c.y - b.y).toDouble(), (c.x - b.x).toDouble()) -
                    Math.atan2((a.y - b.y).toDouble(), (a.x - b.x).toDouble())
            var angle = Math.abs(radians * 180.0 / Math.PI)
            if (angle > 180.0) {
                angle = 360.0 - angle
            }
            return angle
        }
    }
}

// --- DATA CLASSES ---
// Ensure you do NOT have a BoundingBox.kt file elsewhere, or delete it!
data class BoundingBox(
    val x1: Float, val y1: Float, val x2: Float, val y2: Float,
    val cx: Float, val cy: Float, val w: Float, val h: Float,
    val cnf: Float, val cls: Int, val clsName: String,
    val keypoints: List<Keypoint>
)

data class Keypoint(
    val x: Float, val y: Float, val confidence: Float
)

// Calculate the angle formed by three points: A (Anchor), B (Pivot), C (End)
// Example: Hip(A) -> Knee(B) -> Ankle(C)
fun calculateAngle(a: Keypoint, b: Keypoint, c: Keypoint): Double {
    val radians = Math.atan2((c.y - b.y).toDouble(), (c.x - b.x).toDouble()) -
            Math.atan2((a.y - b.y).toDouble(), (a.x - b.x).toDouble())

    var angle = Math.abs(radians * 180.0 / Math.PI)

    if (angle > 180.0) {
        angle = 360.0 - angle
    }

    return angle
}
package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results = listOf<BoundingBox>()

    // 1. Define the Paints (The Look)
    private val pointPaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.FILL
        alpha = 200
    }

    private val linePaint = Paint().apply {
        color = Color.CYAN // Default color (will change dynamically later)
        style = Paint.Style.STROKE
        strokeWidth = 8f
        strokeCap = Paint.Cap.ROUND // Smooth rounded line ends
        alpha = 180
    }

    // 2. Define the Body Connections (COCO Format)
    // pairs of indices: e.g., 5-7 is Left Shoulder to Left Elbow
    private val bodyConnections = listOf(
        Pair(5, 7), Pair(7, 9),       // Left Arm
        Pair(6, 8), Pair(8, 10),      // Right Arm
        Pair(5, 6),                   // Shoulders
        Pair(5, 11), Pair(6, 12),     // Torso
        Pair(11, 12),                 // Hips
        Pair(11, 13), Pair(13, 15),   // Left Leg
        Pair(12, 14), Pair(14, 16)    // Right Leg
    )

    fun setResults(boundingBoxes: List<BoundingBox>) {
        results = boundingBoxes
        invalidate()
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        for (result in results) {
            val keypoints = result.keypoints

            // 3. Draw the Skeleton (Lines)
            for (connection in bodyConnections) {
                val start = keypoints[connection.first]
                val end = keypoints[connection.second]

                // Only draw if both points are confident
                if (start.confidence > 0.5 && end.confidence > 0.5) {
                    // Convert relative coordinates (0..1) to screen pixels
                    val startX = start.x * width
                    val startY = start.y * height
                    val endX = end.x * width
                    val endY = end.y * height

                    canvas.drawLine(startX, startY, endX, endY, linePaint)
                }
            }

            // 4. Draw the Joints (Dots)
            for (keypoint in keypoints) {
                if (keypoint.confidence > 0.5) {
                    val px = keypoint.x * width
                    val py = keypoint.y * height
                    // Draw a glow effect (optional) or just the dot
                    canvas.drawCircle(px, py, 10f, pointPaint)
                }
            }
        }
    }
}
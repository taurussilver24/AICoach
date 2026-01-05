package com.surendramaran.yolov8tflite

import kotlin.math.abs
import kotlin.math.atan2

object PoseUtils {

    // 1. THE MATH: Calculate angle between three joints
    fun calculateAngle(a: Keypoint, b: Keypoint, c: Keypoint): Double {
        val radians = atan2((c.y - b.y).toDouble(), (c.x - b.x).toDouble()) -
                atan2((a.y - b.y).toDouble(), (a.x - b.x).toDouble())
        var angle = abs(radians * 180.0 / Math.PI)
        if (angle > 180.0) angle = 360.0 - angle
        return angle
    }

    // 2. THE LOGIC: Analyze a list of frames to find reps and errors
    fun analyzeSession(frames: List<FrameData>): WorkoutSummary {
        var state = "UP"
        var repCount = 0
        var deepReps = 0

        // Loop through the recorded history
        for (frame in frames) {
            val angle = frame.angle // Changed from primaryAngle to angle (matches FrameData)

            // Simple State Machine for Squats
            if (state == "UP" && angle < 140) {
                state = "DOWN"
            }
            else if (state == "DOWN") {
                if (angle > 160) {
                    // Rep completed
                    state = "UP"
                    repCount++
                    // In a real app, you'd check the MIN angle of the rep
                    // For now, we assume if they went down, it counts
                    deepReps++
                }
            }
        }

        // FIX IS HERE: Keep it as Float (removed .toInt())
        val accuracy = if (repCount > 0) (deepReps.toFloat() / repCount * 100) else 0f

        val feedbackString = "User performed $repCount reps with ${accuracy.toInt()}% perfect depth."

        return WorkoutSummary(
            totalReps = repCount,
            accuracy = accuracy, // Now passing Float to Float (Correct)
            aiPrompt = feedbackString
        )
    }
}

// Data Classes (Ensure these match MainActivity)
data class FrameData(val timestamp: Long, val keypoints: List<Keypoint>, val angle: Double)
data class WorkoutSummary(val totalReps: Int, val accuracy: Float, val aiPrompt: String)
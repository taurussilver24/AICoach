package com.surendramaran.yolov8tflite

import android.content.Context
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

class CoachBrain(private val context: Context) {
    private var llmInference: LlmInference? = null

    suspend fun initialize() {
        withContext(Dispatchers.IO) {
            val modelFile = File("/data/local/tmp/llm_model.bin")
            if (modelFile.exists()) {
                val options = LlmInference.LlmInferenceOptions.builder()
                    .setModelPath(modelFile.absolutePath)
                    .setMaxTokens(1024) // Allow longer, smarter answers
                    .setTopK(40)
                    .setTemperature(0.8f) // A bit more creative
                    .build()
                llmInference = LlmInference.createFromOptions(context, options)
            }
        }
    }

    suspend fun generateAnalysis(reps: Int, score: Int, exercise: String): String {
        return withContext(Dispatchers.IO) {
            if (llmInference == null) return@withContext "Coach is initializing... please wait."

            // Smart Logic: Determine the 'Vibe' based on the score
            val vibe = if (score > 85) "Elite Athlete" else if (score > 60) "Intermediate" else "Beginner"

            // The "Intelligent" Prompt formatted specifically for Gemma-IT
            val prompt = """
                <start_of_turn>user
                Role: You are an elite, biomechanically expert fitness coach.
                Task: Analyze the user's latest set.
                
                Data:
                - Exercise: $exercise
                - Reps Performed: $reps
                - Form Accuracy Score: $score% (Level: $vibe)
                
                Output Format (Strictly follow this):
                Feedback: [1 sentence summarizing performance]
                Tip: [1 specific biomechanical correction based on the score]
                Next: [1 motivational command]
                
                Do not use markdown symbols like hashtags. Keep it professional.<end_of_turn>
                <start_of_turn>model
            """.trimIndent()

            try {
                llmInference!!.generateResponse(prompt)
            } catch (e: Exception) {
                "Analysis failed: ${e.message}"
            }
        }
    }
}
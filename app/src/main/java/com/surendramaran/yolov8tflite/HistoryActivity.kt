package com.surendramaran.yolov8tflite

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import com.surendramaran.yolov8tflite.databinding.ActivityHistoryBinding
import java.util.concurrent.Executors

class HistoryActivity : AppCompatActivity() {
    private lateinit var binding: ActivityHistoryBinding
    private lateinit var adapter: HistoryAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHistoryBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Setup RecyclerView
        binding.recyclerView.layoutManager = LinearLayoutManager(this)
        adapter = HistoryAdapter(emptyList())
        binding.recyclerView.adapter = adapter

        binding.backButton.setOnClickListener { finish() }

        loadHistory()
    }

    private fun loadHistory() {
        // Run database query in background thread
        Executors.newSingleThreadExecutor().execute {
            val db = AppDatabase.getDatabase(this)
            val history = db.workoutDao().getAll()

            runOnUiThread {
                adapter.updateData(history)
            }
        }
    }
}
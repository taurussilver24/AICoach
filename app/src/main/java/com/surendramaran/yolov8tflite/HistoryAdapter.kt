package com.surendramaran.yolov8tflite

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import java.text.SimpleDateFormat
import java.util.*

class HistoryAdapter(private var sessions: List<WorkoutSession>) :
    RecyclerView.Adapter<HistoryAdapter.ViewHolder>() {

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val tvDate: TextView = view.findViewById(R.id.tvDate)
        val tvScore: TextView = view.findViewById(R.id.tvScore)
        val tvStats: TextView = view.findViewById(R.id.tvStats)
        val tvFeedback: TextView = view.findViewById(R.id.tvFeedback)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_history, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val session = sessions[position]

        // Format Date
        val sdf = SimpleDateFormat("MMM dd, HH:mm", Locale.getDefault())
        holder.tvDate.text = sdf.format(Date(session.timestamp))

        // Color Code Score
        holder.tvScore.text = "${session.formScore}%"
        holder.tvScore.setTextColor(if(session.formScore >= 80) 0xFF4CAF50.toInt() else 0xFFFFC107.toInt())

        holder.tvStats.text = "${session.exerciseType} • ${session.reps} Reps • ${session.durationSec}s"
        holder.tvFeedback.text = "Coach: ${session.feedback}"
    }

    override fun getItemCount() = sessions.size

    fun updateData(newSessions: List<WorkoutSession>) {
        sessions = newSessions
        notifyDataSetChanged()
    }
}
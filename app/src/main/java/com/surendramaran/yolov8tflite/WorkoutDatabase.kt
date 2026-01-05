package com.surendramaran.yolov8tflite

import android.content.Context
import androidx.room.*

// 1. The Table (Entity)
@Entity(tableName = "workout_history")
data class WorkoutSession(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val timestamp: Long,
    val exerciseType: String,
    val reps: Int,
    val formScore: Int,
    val durationSec: Long,
    val feedback: String // "Great job!"
)

// 2. The Access Object (DAO)
@Dao
interface WorkoutDao {
    @Insert
    fun insert(session: WorkoutSession)

    @Query("SELECT * FROM workout_history ORDER BY timestamp DESC")
    fun getAll(): List<WorkoutSession>
}

// 3. The Database
@Database(entities = [WorkoutSession::class], version = 1)
abstract class AppDatabase : RoomDatabase() {
    abstract fun workoutDao(): WorkoutDao

    companion object {
        // Singleton pattern to prevent multiple DB instances
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "fitness_db"
                ).build()
                INSTANCE = instance
                instance
            }
        }
    }
}
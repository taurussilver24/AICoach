// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    // Add "apply false" to both of these lines
    id("com.android.application") version "8.4.0" apply false
    id("com.android.library") version "8.4.0" apply false

    // (Other plugins like kotlin-android might be here too, keep them "apply false" as well)
    id("org.jetbrains.kotlin.android") version "1.9.0" apply false
}

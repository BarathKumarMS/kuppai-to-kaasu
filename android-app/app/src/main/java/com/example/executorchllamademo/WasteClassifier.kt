package com.example.executorchllamademo

import android.content.ContentResolver
import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

data class ClassifyResult(
    val label: String,
    val confidence: Float,
    val scores: List<Pair<String, Float>>
)

/**
 * On-device waste classifier. Loads waste_classifier.tflite and labels.txt from assets.
 * Input: 256 x 256 RGB, float32, raw pixel values 0-255 (scaling is inside the model).
 */
class WasteClassifier(context: Context) {

    private val labels: List<String> =
        context.assets.open(LABELS_FILE).bufferedReader().use { reader ->
            reader.readLines().map { it.trim() }.filter { it.isNotEmpty() }
        }

    private val interpreter: Interpreter =
        Interpreter(loadModel(context), Interpreter.Options().apply { setNumThreads(4) })

    private fun loadModel(context: Context): MappedByteBuffer {
        val fd = context.assets.openFd(MODEL_FILE)
        val buffer = FileInputStream(fd.fileDescriptor).use { stream ->
            stream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
        fd.close()
        return buffer
    }

    fun classify(source: Bitmap): ClassifyResult {
        val bmp = if (source.width == SIZE && source.height == SIZE) {
            source
        } else {
            Bitmap.createScaledBitmap(source, SIZE, SIZE, true)
        }

        val input = ByteBuffer.allocateDirect(4 * SIZE * SIZE * 3).order(ByteOrder.nativeOrder())
        val pixels = IntArray(SIZE * SIZE)
        bmp.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        for (p in pixels) {
            input.putFloat(((p shr 16) and 0xFF).toFloat()) // R
            input.putFloat(((p shr 8) and 0xFF).toFloat())  // G
            input.putFloat((p and 0xFF).toFloat())          // B
        }
        input.rewind()

        val output = Array(1) { FloatArray(labels.size) }
        interpreter.run(input, output)

        val scores = labels.indices
            .map { labels[it] to output[0][it] }
            .sortedByDescending { it.second }
        return ClassifyResult(scores[0].first, scores[0].second, scores)
    }

    companion object {
        const val SIZE = 256
        private const val MODEL_FILE = "waste_classifier.tflite"
        private const val LABELS_FILE = "labels.txt"

        /** Decodes a photo straight to 256 x 256 (software bitmap, EXIF rotation handled). */
        fun loadBitmap(resolver: ContentResolver, uri: Uri): Bitmap {
            val source = ImageDecoder.createSource(resolver, uri)
            return ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.setTargetSize(SIZE, SIZE)
            }
        }
    }
}

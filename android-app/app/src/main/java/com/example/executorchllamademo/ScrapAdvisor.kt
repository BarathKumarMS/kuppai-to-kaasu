package com.example.executorchllamademo

import android.content.Context
import org.json.JSONObject

/**
 * Reads scrap_rates.json and prompt_template.txt from assets and builds the
 * text that is sent to the on-device Llama model.
 */
class ScrapAdvisor(context: Context) {

    private val root = JSONObject(readAsset(context, "scrap_rates.json"))
    private val template = readAsset(context, "prompt_template.txt")

    val threshold: Float = root.optDouble("confidence_threshold", 0.60).toFloat()

    fun defaultQuestion(): String =
        "How should I prepare this for recycling, and is it worth selling to a scrap dealer?"

    private fun classInfo(label: String): JSONObject? =
        root.optJSONObject("classes")?.optJSONObject(label)

    fun displayName(label: String): String =
        classInfo(label)?.optString("display_name", label) ?: label

    private fun percent(confidence: Float): Int = Math.round(confidence * 100)

    fun detectionLine(r: ClassifyResult): String {
        val info = classInfo(r.label) ?: return "Detected: ${r.label} (${percent(r.confidence)}% sure)."
        return "Detected: ${displayName(r.label)} (${percent(r.confidence)}% sure). " +
            "Approx. scrap value: Rs ${info.optInt("min")} to Rs ${info.optInt("max")} per kg."
    }

    fun notSureMessage(r: ClassifyResult): String {
        val base = root.optJSONObject("fallback")?.optString("message")
            ?: "I am not sure what this is. Try another angle with good light."
        return "$base (best guess: ${displayName(r.label)}, ${percent(r.confidence)}%)"
    }

    fun buildPrompt(r: ClassifyResult, question: String): String {
        val info = classInfo(r.label) ?: return question

        val prep = StringBuilder(info.optString("prep"))
        val subtypes = info.optJSONArray("subtypes")
        if (subtypes != null && subtypes.length() > 1) {
            prep.append(" Typical rates per kg: ")
            for (i in 0 until subtypes.length()) {
                val s = subtypes.getJSONObject(i)
                if (i > 0) prep.append("; ")
                prep.append("${s.optString("name")} Rs ${s.optInt("min")}-${s.optInt("max")}")
            }
            prep.append(".")
        }

        return template
            .replace("{display_name}", displayName(r.label))
            .replace("{confidence}", percent(r.confidence).toString())
            .replace("{min}", info.optInt("min").toString())
            .replace("{max}", info.optInt("max").toString())
            .replace("{typical}", info.optInt("typical").toString())
            .replace("{prep}", prep.toString())
            .replace("{question}", question)
    }

    companion object {
        private fun readAsset(context: Context, name: String): String =
            context.assets.open(name).bufferedReader().use { it.readText() }
    }
}

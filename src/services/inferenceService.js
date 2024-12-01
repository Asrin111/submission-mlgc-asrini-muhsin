const tf = require("@tensorflow/tfjs-node");
const InputError = require("../exceptions/InputError");

async function predictClassification(model, image) {
  try {
    const tensor = tf.node
      .decodeJpeg(image)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat();

    const prediction = model.predict(tensor);
    const score = await prediction.array();  // Mendapatkan array hasil prediksi
    const confidenceScore = Math.max(...score[0]) * 100; // Menyimpan nilai probabilitas tertinggi

    // Menentukan hasil klasifikasi berdasarkan probabilitas
    let result;
    let suggestion;

    // Ambil probabilitas dari hasil prediksi
    const cancerProb = score[0][0];  // Asumsi cancer adalah kelas pertama (index 0)
    const nonCancerProb = score[0][1];  // Asumsi non-cancer adalah kelas kedua (index 1)

    if (cancerProb > 0.5) {
      result = "Cancer";
      suggestion = "Segera periksa ke dokter!";
    } else {
      result = "Non-cancer";
      suggestion = "Penyakit kanker tidak terdeteksi.";
    }

    return { result, suggestion, confidenceScore };
  } catch (error) {
    throw new InputError("Terjadi kesalahan dalam melakukan prediksi");
  }
}

module.exports = predictClassification;

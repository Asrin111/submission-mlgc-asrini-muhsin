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
    const score = await prediction.array();
    const confidenceScore = Math.max(...score[0]) * 100;

    const classes = [
      "Melanocytic nevus",
      "Squamous cell carcinoma",
      "Vascular lesion",
    ];
    const classResult = tf.argMax(prediction, 1).dataSync()[0];
    const label = classes[classResult];

    let suggestion;
    let result;

    if (label === "Melanocytic nevus" || label === "Squamous cell carcinoma") {
      result = "Cancer";
      suggestion = "Segera periksa ke dokter!";
    } else if (label === "Vascular lesion") {
      result = "Non-cancer";
      suggestion = "Anda sehat!";
    }

    return { result, suggestion, confidenceScore };
  } catch (error) {
    throw new InputError("Terjadi kesalahan dalam melakukan prediksi");
  }
}

module.exports = predictClassification;

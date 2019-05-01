let model;
let chart;

tf.loadLayersModel("./model.json").then((mod) => {
    model = mod;
});

function setup() {
    createCanvas(280, 280);
    background(0);
}

function clearCanvas() {
    background(0);
}

function mouseDragged() {
    noStroke();
    fill(255);
    ellipse(mouseX, mouseY, 15, 15);
}

function mouseReleased() {
    const canvas = document.getElementById("defaultCanvas0");
    let ctx = canvas.getContext("2d");
    let imageData = ctx.getImageData(0, 0, 280, 280);
    model
        .predict(preprocessCanvas(imageData))
        .data()
        .then((predictions) => {
            predictions = concatenate(predictions, categories);
            predictions = predictions
                .sort((a, b) => {
                    return b.a - a.a;
                })
                .slice(0, 5);
            const canvas = document.getElementById("barChart");
            const ctx = canvas.getContext("2d");
            if (chart) {
                chart.destroy();
            }
            const labels = [];
            const data = [];
            const colors = [];
            for (const p of predictions) {
                labels.push(p.b);
                data.push((p.a * 100).toFixed(3));
                colors.push("hsl(" + floor(random(360)) + ",100%,40%)");
            }
            console.log(labels);

            let datal = {
                labels: labels,
                datasets: [
                    {
                        fill: true,
                        backgroundColor: colors,
                        data: data
                    }
                ]
            };
            chart = new Chart(ctx, {
                type: "pie",
                data: datal,
                options: {
                    responsive: false,
                    maintainAspectRatio: false
                }
            });
        });
}

preprocessCanvas = (canvas) => {
    // Preprocess image for the network
    let tensor = tf.browser
        .fromPixels(canvas) // Shape: (300, 300, 3) - RGB image
        .resizeNearestNeighbor([28, 28]) // Shape: (28, 28, 3) - RGB image
        .mean(2) // Shape: (28, 28) - grayscale
        .expandDims(2) // Shape: (28, 28, 1) - network expects 3d values with channels in the last dimension
        .expandDims() // Shape: (1, 28, 28, 1) - network makes predictions for "batches" of images
        .toFloat(); // Network works with floating points inputs
    return tensor.div(255.0); // Normalize [0..255] values into [0..1] range
};

function concatenate(array1, array2) {
    if (array1.length !== array2.length) {
        console.error("the 2 arrays needs to be the same length");
        return;
    }
    const result = [];
    for (const i in array1) {
        result.push({ a: array1[i], b: array2[i] });
    }
    return result;
}

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const startButton = document.getElementById("startButton");
const verifyButton = document.getElementById("verifyButton"); // Add this line
const messageElement = document.getElementById("message"); // Add this line for displaying messages
const verifiying = document.getElementById("verifiying");


let ctx;
let videoWidth, videoHeight;
let model; // Define the model variable
let stopRendering = true; // Initialize the rendering flag

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            videoWidth = video.videoWidth;
            videoHeight = video.videoHeight;
            video.width = videoWidth;
            video.height = videoHeight;
            resolve(video);
        };
    });
}

async function setupCanvas() {
    canvas.width = videoWidth;
    canvas.height = videoHeight;

    ctx = canvas.getContext('2d');
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);

    ctx.fillStyle = "green";
}



function hideStartButton() {
    $("#startButton").hide();
    // $("#verifyButton").show();
    // $("#verifying").show();
}
async function loadFaceLandmarkDetectionModel() {
    return faceLandmarksDetection
        .load(faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
            { maxFaces: 1 });
}

const CONFIDENCE_THRESHOLD = 0.7;
const PIN_POINTING = 0.4;
const DETECTED = 1;


async function captureFrame( width, height) {
    ctx = canvas.getContext("2d");
    // canvas.width = width;
    // canvas.height = height;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageBlob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg"));
    console.log(imageBlob);
    return imageBlob;
}

async function renderPrediction() {
    const predictions = await model.estimateFaces({
        input: video,
        returnTensors: false,
        flipHorizontal: false,
        predictIrises: false
    });

    ctx.drawImage(
        video, 0, 0, video.width, video.height, 0, 0, canvas.width, canvas.height,);
        
    
    

    if (predictions.length > 0 && !stopRendering) {
       
        predictions.forEach(prediction => {
            var faceInViewConfidence = prediction.faceInViewConfidence;
            if(faceInViewConfidence > CONFIDENCE_THRESHOLD){ 
                     
                const scaledMesh = prediction.scaledMesh;
                for (let i = 0; i < scaledMesh.length; i++){
                    
                    const x = scaledMesh[i][0];
                    const y = scaledMesh[i][1];
                    ctx.beginPath();
                    ctx.arc(x, y, 2, 0, 2 * Math.PI);
                    ctx.fill();
                    
                    
                }
                console.log("Pin pointing face level" + faceInViewConfidence);
                setTimeout(() => {
                    stopRendering = true; 
                    console.log("Face Captured.");
                    // window.location.href = "old_index.html";
                    // captureFrame(video.width, video.height);
                }, 5000);
            }
          
            // else if ( faceInViewConfidence == DETECTED){
            //     console.log("Face Detected.");
              
            //     setTimeout(() => {
            //         stopRendering = true; 
            //         console.log("Face Captured.");
            //         // window.location.href = "old_index.html";
            //         // captureFrame(video.width, video.height);
            //     }, 5000);
               

            // }
            //  else{
            // //     console.log("please posistion your for detections");
            // }

        });
    } else{
        console.log("no detection")
    }

    if (!stopRendering) {
        window.requestAnimationFrame(renderPrediction);
    }
    
}

startButton.addEventListener("click", async () => {
    // Set up camera
    hideStartButton();
    await setupCamera();

    // Set up canvas
    await setupCanvas();

    // Load the model
    model = await loadFaceLandmarkDetectionModel();

    // Start rendering Face Mesh Prediction
    stopRendering = false;
    renderPrediction();
});

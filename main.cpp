#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <map>
#include <vector>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace dnn;
using namespace std;

// Structure to hold vehicle information
struct Vehicle {
    int id;
    KalmanFilter kf;
    Point2f position;
    Rect boundingBox;
    double stoppedTime;
};

// Function to initialize a Kalman filter for a new vehicle
KalmanFilter createKalmanFilter(Point2f initialPos) {
    KalmanFilter kf(4, 2, 0);
    kf.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
    kf.statePre.at<float>(0) = initialPos.x;
    kf.statePre.at<float>(1) = initialPos.y;
    kf.statePre.at<float>(2) = 0;
    kf.statePre.at<float>(3) = 0;
    kf.measurementMatrix = (Mat_<float>(2, 4) <<
                            1, 0, 0, 0,
                            0, 1, 0, 0);
    setIdentity(kf.processNoiseCov, Scalar::all(1e-2));
    setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(kf.errorCovPost, Scalar::all(1));
    return kf;
}

// Function to update vehicle position using Kalman filter
void updateKalmanFilter(Vehicle &vehicle, Point2f measurement) {
    Mat prediction = vehicle.kf.predict();
    Mat_<float> measurementMat(2, 1);
    measurementMat(0) = measurement.x;
    measurementMat(1) = measurement.y;
    Mat estimated = vehicle.kf.correct(measurementMat);
    vehicle.position = Point2f(estimated.at<float>(0), estimated.at<float>(1));
}

// Function to detect stopped vehicles
bool detectStoppedVehicles(int id , Vehicle &vehicle, double currentTime, double stopTimeThreshold, double movementThreshold , map<int, double>& stoppedTimes) {
    // Calculate the distance moved by the vehicle
    float previousPosX = vehicle.kf.statePre.at<float>(0);
    float previousPosY = vehicle.kf.statePre.at<float>(1);
    Point2f previousPos(previousPosX, previousPosY);
    double distanceMoved = norm(vehicle.position - vehicle.kf.statePre.at<Point2f>(0));
    cout<<distanceMoved<<endl;
    if (distanceMoved < movementThreshold) {
        double startTime = stoppedTimes[id];
        // If the vehicle has moved less than the movement threshold
        if (stoppedTimes.find(id) == stoppedTimes.end()) {
            // If the vehicle is not in the stopped vehicles map, update its stopped time
            stoppedTimes[id] = currentTime;
        } else if (currentTime - startTime >= stopTimeThreshold) {
            stoppedTimes[id] = currentTime;
            // If the vehicle has been stopped for the stop time threshold
            cout << "Vehicle ID: " << vehicle.id << " - Detected as stopped." << endl;
            // Debugging: Print distance moved, current time, etc.
            cout << "Distance Moved: " << distanceMoved << endl;
            cout << "Current Time: " << currentTime << endl;
            return true; // Vehicle is stopped
        }
    } else {
        // Reset the stopped time if the vehicle moves
        if (stoppedTimes.find(id) != stoppedTimes.end()) {
            stoppedTimes.erase(id);
        }
    }
    
    // Debugging: Print vehicle ID, distance moved, current time, stopped time, etc.
    return false; // Vehicle is not stopped
}

int main() {
    

    string videoPath = "/Users/krisanusarkar/Documents/ML/intoziproject/stopped_vehicle_assignment.avi";
    string outputPath = "your_name_stopped_vehicle_result.avi";
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    

    // Load YOLO model
    string modelConfiguration = "/Users/krisanusarkar/Documents/ML/intoziproject/YoloV3.cfg";
    string modelWeights = "/Users/krisanusarkar/Documents/ML/intoziproject/yolov3.weights";
    string classFile = "/Users/krisanusarkar/Documents/ML/intoziproject/Coco.names";
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    double stopTimeThreshold = 2.0; // 5 seconds threshold
    double movementThreshold = 50.0; // Distance threshold to consider a vehicle stopped
    int nextVehicleID = 1;
    double currentTime = 0.0;
    double frameRate = cap.get(CAP_PROP_FPS);
    double frameTime = 1.0 /frameRate;

    map<int, Vehicle> vehicles;
    map<int, double> stoppedTimes;

    // Create a VideoWriter object to save the output video
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = static_cast<int>(cap.get(CAP_PROP_FPS));
    cout<<frame_height<<" "<<frame_height<<endl ;
    VideoWriter outputVideo("your_name_stopped_vehicle_result.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(frame_width, frame_height));

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        currentTime += frameTime;
        vehicles.clear();

        // Detect vehicles using YOLO
        Mat blob = dnn::blobFromImage(frame, 1/255.0, Size(640 , 640), Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        vector<Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        vector<Rect> detectedBoxes;
        vector<Point2f> detectedCenters;
        vector<int> classIds;
        vector<float> confidences;
        for (size_t i = 0; i < outs.size(); ++i) {
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint); // ???
                if (confidence > 0.5  && classIdPoint.x == 2) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    detectedBoxes.push_back(Rect(left, top, width, height));
                    detectedCenters.push_back(Point2f(centerX, centerY));
                    confidences.push_back((float)confidence);
                }
            }
        }

        vector<int> indices;
        NMSBoxes(detectedBoxes, confidences , 0.5, 0.4, indices);

        // Filter out bounding boxes based on NMS indices
        vector<Rect> filteredBoxes;
        for (int i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            filteredBoxes.push_back(detectedBoxes[idx]);
        }       

        // Match detected boxes with existing vehicles using nearest neighbor
        for (size_t i = 0; i < filteredBoxes.size(); i++) {
            Point2f center = detectedCenters[i];
            Rect box = filteredBoxes[i];

            double minDist = DBL_MAX;
            int bestMatchID = -1;

            for (auto& vehiclePair : vehicles) {
                int id = vehiclePair.first;
                Vehicle& vehicle = vehiclePair.second;

                double dist = norm(vehicle.position - center);
                if (dist < minDist) {
                    minDist = dist;
                    bestMatchID = id;
                }
            }

            if (minDist < movementThreshold) {
                // Update existing vehicle
                Vehicle& vehicle = vehicles[bestMatchID];
                updateKalmanFilter(vehicle, center);
                vehicle.boundingBox = box;
            } else {
                // Add new vehicle
                Vehicle newVehicle;
                newVehicle.id = nextVehicleID++;
                newVehicle.kf = createKalmanFilter(center);
                newVehicle.position = center;
                newVehicle.boundingBox = box;
                newVehicle.stoppedTime = 0 ;
                vehicles[newVehicle.id] = newVehicle;
            }
        }

        // Draw bounding boxes and detect stopped vehicles
        for (auto& vehiclePair : vehicles) {
            int id = vehiclePair.first;
            Vehicle& vehicle = vehiclePair.second;

            bool isStopped = detectStoppedVehicles(id , vehicle, currentTime, stopTimeThreshold, movementThreshold , stoppedTimes);
            Scalar color = isStopped ? Scalar(0, 0, 255) : Scalar(0, 255, 0);
            rectangle(frame, vehicle.boundingBox, color, 2);
            putText(frame, format("ID: %d", id), vehicle.boundingBox.tl(), FONT_HERSHEY_SIMPLEX, 0.5, color, 2);;
        }

        outputVideo.write(frame);
    }

    cap.release();
    outputVideo.release();

    return 0;
}

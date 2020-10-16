#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, char** argv) {
	Mat src = imread("kong3.jpg", IMREAD_COLOR);

	if (src.empty()) {
		cerr << "이미지 로드 실패" << endl;
		return -1;
	}


	// ORB feature 생성
	Ptr<Feature2D> feature = ORB::create();

	// KeyPoint단위로 벡터 생성 
	vector<KeyPoint> keypoints;
	// source 이미지의 keypoint를 detect(ORB의 feature로)
	feature->detect(src, keypoints);

	//descriptor 객체 생성
	Mat desc;
	feature->compute(src, keypoints, desc);

	cout << "keypoints size :" << keypoints.size() << endl;
	cout << "descriptor size : " << desc.size() << endl;

	// output 이미지 객체 생성
	Mat out;

	// 주성분 방향 표시하려면 DRAW_RICH_KEYPOINTS
	drawKeypoints(src, keypoints, out);

	imshow("src", src);
	imshow("out", out);

	waitKey();

	return 0;

}
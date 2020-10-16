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
		cerr << "�̹��� �ε� ����" << endl;
		return -1;
	}


	// ORB feature ����
	Ptr<Feature2D> feature = ORB::create();

	// KeyPoint������ ���� ���� 
	vector<KeyPoint> keypoints;
	// source �̹����� keypoint�� detect(ORB�� feature��)
	feature->detect(src, keypoints);

	//descriptor ��ü ����
	Mat desc;
	feature->compute(src, keypoints, desc);

	cout << "keypoints size :" << keypoints.size() << endl;
	cout << "descriptor size : " << desc.size() << endl;

	// output �̹��� ��ü ����
	Mat out;

	// �ּ��� ���� ǥ���Ϸ��� DRAW_RICH_KEYPOINTS
	drawKeypoints(src, keypoints, out);

	imshow("src", src);
	imshow("out", out);

	waitKey();

	return 0;

}
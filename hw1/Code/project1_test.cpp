#include <cstring>
#include <QtGui>
#include <QtTest/QtTest>

#include "mainwindow.h"

class Project1Test : public QObject {
  Q_OBJECT
private slots:
  void TestBlackWhiteImage();
  void TestConvolution();
  void TestGaussianBlur();
  void TestFirstDerivImage_x();
  void TestFirstDerivImage_y();
  void TestSecondDerivImage();
  void TestSharpenImage();
  void TestSobelImage();
  void TestRotateImage();
  void TestFindPeaksImage();
  void TestRandomSeedImage();
  void TestPixelSeedImage();
};

namespace {
class TestWindow : private MainWindow {
  friend class ::Project1Test;
};
}  // namespace

void Project1Test::TestBlackWhiteImage() {
  QImage image(1, 1, QImage::Format_RGB32);
  for (int i = 0; i < image.height(); ++i)
    for (int j = 0; j < image.width(); ++j)
      image.setPixel(i, j, qRgb(100, 200, 50));

  TestWindow window;
  window.BlackWhiteImage(&image);

  QRgb pixel = image.pixel(0, 0);
  QCOMPARE(pixel, qRgb(155, 155, 155));
}

void Project1Test::TestConvolution() {
  TestWindow window;
  window.imageHeight = 4;
  window.imageWidth = 3;
  double **image = new double*[12]{
    new double[3]{1, 2, 3}, new double[3]{4, 5, 6}, new double[3]{7, 8, 9},
    new double[3]{10, 11, 12}, new double[3]{13, 14, 15}, new double[3]{16, 17, 18},
    new double[3]{19, 20, 21}, new double[3]{22, 23, 24}, new double[3]{25, 26, 27},
    new double[3]{28, 29, 30}, new double[3]{31, 32, 33}, new double[3]{34, 35, 36}};
  double kernel[9] = {1, 2, 3,
                      4, 5, 4,
                      3, 2, 1};
  window.Convolution(image, kernel, 3, 3, false);  
  QCOMPARE(image[0][0], 5*1 + 4*4 + 2*10 + 1*13);
  QCOMPARE(image[0][1], 5*2 + 4*5 + 2*11 + 1*14);
  QCOMPARE(image[0][2], 5*3 + 4*6 + 2*12 + 1*15);

  QCOMPARE(image[11][0], 1*22 + 2*25 + 4*31 + 5*34);
  QCOMPARE(image[11][1], 1*23 + 2*26 + 4*32 + 5*35);
  QCOMPARE(image[11][2], 1*24 + 2*27 + 4*33 + 5*36);

  for (int i = 0; i < window.imageHeight*window.imageWidth; ++i) delete[] image[i];
  delete[] image;
}

void Project1Test::TestGaussianBlur() {
  TestWindow window;
  window.imageHeight = 4;
  window.imageWidth = 3;

  double **image1 = new double*[12]{
    new double[3]{1, 2, 3}, new double[3]{4, 5, 6}, new double[3]{7, 8, 9},
    new double[3]{10, 11, 12}, new double[3]{13, 14, 15}, new double[3]{16, 17, 18},
    new double[3]{19, 20, 21}, new double[3]{22, 23, 24}, new double[3]{25, 26, 27},
    new double[3]{28, 29, 30}, new double[3]{31, 32, 33}, new double[3]{34, 35, 36}};

  double **image2 = new double*[12];
  for (int i = 0; i < window.imageHeight*window.imageWidth; ++i) {
    image2[i] = new double[3];
    std::memcpy(image2[i], image1[i], 3*sizeof(double));
  }

  window.GaussianBlurImage(image1, 2.);
  window.SeparableGaussianBlurImage(image2, 2.);

  for (int i = 0; i < window.imageHeight*window.imageWidth; ++i)
    for (int j = 0; j < 3; ++j)
      QVERIFY(qFloatDistance(image1[i][j], image2[i][j]) < (1ULL << 34));

  for (int i = 0; i < window.imageHeight*window.imageWidth; ++i) {
    delete[] image1[i]; delete[] image2[i];
  }
  delete[] image1; delete[] image2;
}

void Project1Test::TestFirstDerivImage_x() {
  TestWindow window;
  window.imageHeight = 4;
  window.imageWidth = 3;

  double **image = new double*[12]{
    new double[3]{1, 2, 3}, new double[3]{4, 5, 6}, new double[3]{7, 8, 9},
    new double[3]{10, 11, 12}, new double[3]{13, 14, 15}, new double[3]{16, 17, 18},
    new double[3]{19, 20, 21}, new double[3]{22, 23, 24}, new double[3]{25, 26, 27},
    new double[3]{28, 29, 30}, new double[3]{31, 32, 33}, new double[3]{34, 35, 36}};
  window.FirstDerivImage_x(image, 1.0);

  QCOMPARE(image[4][0], 108.17637380156558890576);
  QCOMPARE(image[4][1], 108.17637380156557469491);
  QCOMPARE(image[4][2], 108.17637380156558890576);
}

void Project1Test::TestFirstDerivImage_y() {
  TestWindow window;
  window.imageHeight = 4;
  window.imageWidth = 3;

  double **image = new double*[12]{
    new double[3]{1, 2, 3}, new double[3]{4, 5, 6}, new double[3]{7, 8, 9},
    new double[3]{10, 11, 12}, new double[3]{13, 14, 15}, new double[3]{16, 17, 18},
    new double[3]{19, 20, 21}, new double[3]{22, 23, 24}, new double[3]{25, 26, 27},
    new double[3]{28, 29, 30}, new double[3]{31, 32, 33}, new double[3]{34, 35, 36}};
  window.FirstDerivImage_y(image, 1.0);

  QCOMPARE(image[4][0], 117.85290332319146955342);
  QCOMPARE(image[4][1], 118.01895729671271340067);
  QCOMPARE(image[4][2], 118.18501127023395724791);
}

void Project1Test::TestSecondDerivImage() {
  TestWindow window;
  window.imageHeight = 4;
  window.imageWidth = 3;

  double **image = new double*[12]{
    new double[3]{1, 2, 3}, new double[3]{4, 5, 6}, new double[3]{7, 8, 9},
    new double[3]{10, 11, 12}, new double[3]{13, 14, 15}, new double[3]{16, 17, 18},
    new double[3]{19, 20, 21}, new double[3]{22, 23, 24}, new double[3]{25, 26, 27},
    new double[3]{28, 29, 30}, new double[3]{31, 32, 33}, new double[3]{34, 35, 36}};
  window.SecondDerivImage(image, 1.0);

  QCOMPARE(image[4][0], 98.72575030383104888188);
  QCOMPARE(image[4][1], 98.01067175307645129578);
  QCOMPARE(image[4][2], 97.29559320232185370969);
}

void Project1Test::TestSharpenImage() {
  TestWindow window;
  window.imageHeight = 4;
  window.imageWidth = 3;

  double **image = new double*[12]{
    new double[3]{1, 2, 3}, new double[3]{4, 5, 6}, new double[3]{7, 8, 9},
    new double[3]{10, 11, 12}, new double[3]{13, 14, 15}, new double[3]{16, 17, 18},
    new double[3]{19, 20, 21}, new double[3]{22, 23, 24}, new double[3]{25, 26, 27},
    new double[3]{28, 29, 30}, new double[3]{31, 32, 33}, new double[3]{34, 35, 36}};
  window.SharpenImage(image, 1.0, 5.0);

  QCOMPARE(image[7][0], 221.4414165503125957);
  QCOMPARE(image[7][1], 226.0168093040856547);
  QCOMPARE(image[7][2], 230.5922020578585148);
}

void Project1Test::TestSobelImage() {
  TestWindow window;
  window.imageHeight = 4;
  window.imageWidth = 3;

  double **image = new double*[12]{
    new double[3]{1, 2, 3}, new double[3]{4, 5, 6}, new double[3]{7, 8, 9},
    new double[3]{10, 11, 12}, new double[3]{13, 14, 15}, new double[3]{16, 17, 18},
    new double[3]{19, 20, 21}, new double[3]{22, 23, 24}, new double[3]{25, 26, 27},
    new double[3]{28, 29, 30}, new double[3]{31, 32, 33}, new double[3]{34, 35, 36}};
  window.SobelImage(image);

  QCOMPARE(image[10][0], 44.4555115728856194);
  QCOMPARE(image[10][1], 26.9555115728856158);
  QCOMPARE(image[10][2], -26.4999999999999964);
}


void Project1Test::TestRotateImage() {
  TestWindow window;
  window.imageHeight = 4;
  window.imageWidth = 3;

  double **image = new double*[12]{
    new double[3]{1, 2, 3}, new double[3]{4, 5, 6}, new double[3]{7, 8, 9},
    new double[3]{10, 11, 12}, new double[3]{13, 14, 15}, new double[3]{16, 17, 18},
    new double[3]{19, 20, 21}, new double[3]{22, 23, 24}, new double[3]{25, 26, 27},
    new double[3]{28, 29, 30}, new double[3]{31, 32, 33}, new double[3]{34, 35, 36}};
  window.RotateImage(image, 20);
}

void Project1Test::TestFindPeaksImage() {
  TestWindow window;
  window.imageHeight = 4;
  window.imageWidth = 3;

  double **image = new double*[12]{
    new double[3]{1, 2, 3}, new double[3]{4, 5, 6}, new double[3]{7, 8, 9},
    new double[3]{10, 11, 12}, new double[3]{13, 14, 15}, new double[3]{16, 17, 18},
    new double[3]{19, 20, 21}, new double[3]{22, 23, 24}, new double[3]{25, 26, 27},
    new double[3]{28, 29, 30}, new double[3]{31, 32, 33}, new double[3]{34, 35, 36}};
  window.FindPeaksImage(image, 40.0);
}

void Project1Test::TestRandomSeedImage() {
  TestWindow window;
  window.imageHeight = 4;
  window.imageWidth = 3;

  double **image = new double*[12]{
    new double[3]{1, 2, 3}, new double[3]{4, 5, 6}, new double[3]{7, 8, 9},
    new double[3]{10, 11, 12}, new double[3]{13, 14, 15}, new double[3]{16, 17, 18},
    new double[3]{19, 20, 21}, new double[3]{22, 23, 24}, new double[3]{25, 26, 27},
    new double[3]{28, 29, 30}, new double[3]{31, 32, 33}, new double[3]{34, 35, 36}};
  window.RandomSeedImage(image, 4);
}
void Project1Test::TestPixelSeedImage() {
  TestWindow window;
  window.imageHeight = 4;
  window.imageWidth = 3;

  double **image = new double*[12]{
    new double[3]{1, 2, 3}, new double[3]{4, 5, 6}, new double[3]{7, 8, 9},
    new double[3]{10, 11, 12}, new double[3]{13, 14, 15}, new double[3]{16, 17, 18},
    new double[3]{19, 20, 21}, new double[3]{22, 23, 24}, new double[3]{25, 26, 27},
    new double[3]{28, 29, 30}, new double[3]{31, 32, 33}, new double[3]{34, 35, 36}};
  window.PixelSeedImage(image, 50);
}

QTEST_MAIN(Project1Test)
#include "project1_test.moc"

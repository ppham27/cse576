#include <QtGui>
#include <QtTest/QtTest>

#include "mainwindow.h"

class Project1Test : public QObject {
  Q_OBJECT
private slots:
  void TestBlackWhiteImage();
  void TestConvolution();
  void TestGaussianBlur();
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

  double **image = new double*[12]{
    new double[3]{1, 2, 3}, new double[3]{4, 5, 6}, new double[3]{7, 8, 9},
    new double[3]{10, 11, 12}, new double[3]{13, 14, 15}, new double[3]{16, 17, 18},
    new double[3]{19, 20, 21}, new double[3]{22, 23, 24}, new double[3]{25, 26, 27},
    new double[3]{28, 29, 30}, new double[3]{31, 32, 33}, new double[3]{34, 35, 36}};

  window.GaussianBlurImage(image, 1.);

  for (int i = 0; i < window.imageHeight*window.imageWidth; ++i) delete[] image[i];
  delete[] image;
}

QTEST_MAIN(Project1Test)
#include "project1_test.moc"

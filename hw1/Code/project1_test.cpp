#include <QtGui>
#include <QtTest/QtTest>

#include "mainwindow.h"

class Project1Test : public QObject {
  Q_OBJECT
private slots:
  void TestBlackWhiteImage();
};

namespace {
class TestWindow : private MainWindow {
private:
  int imageHeight = 32;
  int imageWidth = 32;

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

QTEST_MAIN(Project1Test)
#include "project1_test.moc"

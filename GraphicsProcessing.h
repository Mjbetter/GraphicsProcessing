#pragma once

#ifndef GRAPHICSPROCESSING_H
#define GRAPHICSPROCESSING_H

#include <opencv2\opencv.hpp>
#include "UI.h"

#include <QtWidgets/QMainWindow>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QLabel>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QSlider>
#include <QtGui/QPixmap>
#include <QtGui/QImage>
#include <QtGui/QIcon>
#include <QWidget>
#include <QImage>

class GraphicsProcessing : public QWidget
{
    Q_OBJECT

public:
    GraphicsProcessing(QWidget *parent = nullptr);
    ~GraphicsProcessing();

private slots:
    void openImage();
    void saveImage();
    void applyFilter();

private:
    QImage m_image;

    void loadImage(const QString& fileName);
    void applySepiaFilter();
};
#endif // GRAPHICSPROCESSING_H
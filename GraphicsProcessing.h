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

class GraphicsProcessing : public QMainWindow
{
    Q_OBJECT

public:
    GraphicsProcessing(QWidget *parent = nullptr);

private:
    Ui::GraphicsProcessingClass ui;
    QHBoxLayout* mainLayout;
    QVBoxLayout* toolbarLayout;
    QWidget* toolbarWidget;
    QLabel* toolLabel;
    QPushButton* selectButton;
    QPushButton* drawButton;
    QPushButton* eraseButton;
    QPushButton* fillButton;
    QLabel* brushLabel;
    QSpinBox* brushSizeSpinBox;
    QLabel* opacityLabel;
    QSlider* opacitySlider;
    QLabel* colorLabel;
    QComboBox* colorComboBox;
    QLabel* imageLabel;

    void setupUi();

private slots:
    ;
};
#endif // GRAPHICSPROCESSING_H
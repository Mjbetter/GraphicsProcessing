#pragma once

#include <opencv2\opencv.hpp>
#include "UI.h"


class GraphicsProcessing : public QMainWindow
{
    Q_OBJECT

public:
    GraphicsProcessing(QWidget *parent = nullptr);
    ~GraphicsProcessing();

private:
    Ui::GraphicsProcessingClass ui;
};

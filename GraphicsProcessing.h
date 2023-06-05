#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_GraphicsProcessing.h"

class GraphicsProcessing : public QMainWindow
{
    Q_OBJECT

public:
    GraphicsProcessing(QWidget *parent = nullptr);
    ~GraphicsProcessing();

private:
    Ui::GraphicsProcessingClass ui;
};

#include "GraphicsProcessing.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QFileDialog>
#include <QMessageBox>
#include <QImageReader>
#include <QImageWriter>
#include <QApplication>
#include <QPixmap>

GraphicsProcessing::GraphicsProcessing(QWidget* parent)
    : QWidget(parent)
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);

    QLabel* imageLabel = new QLabel(this);
    imageLabel->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(imageLabel);

    QHBoxLayout* buttonLayout = new QHBoxLayout;

    QPushButton* openButton = new QPushButton(tr("Open"), this);
    connect(openButton, &QPushButton::clicked, this, &GraphicsProcessing::openImage);
    buttonLayout->addWidget(openButton);

    QPushButton* saveButton = new QPushButton(tr("Save"), this);
    connect(saveButton, &QPushButton::clicked, this, &GraphicsProcessing::saveImage);
    buttonLayout->addWidget(saveButton);

    QPushButton* filterButton = new QPushButton(tr("Apply Filter"), this);
    connect(filterButton, &QPushButton::clicked, this, &GraphicsProcessing::applyFilter);
    buttonLayout->addWidget(filterButton);

    mainLayout->addLayout(buttonLayout);
}

GraphicsProcessing::~GraphicsProcessing()
{
}

void GraphicsProcessing::openImage()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Image"), QString(), tr("Images (*.png *.xpm *.jpg)"));
    if (!fileName.isEmpty())
        loadImage(fileName);
}

void GraphicsProcessing::saveImage()
{
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save Image"), QString(), tr("Images (*.png *.xpm *.jpg)"));
    if (!fileName.isEmpty()) {
        QImageWriter writer(fileName);
        if (!writer.write(m_image))
            QMessageBox::information(this, tr("Error"), tr("Could not save image."));
    }
}

void GraphicsProcessing::applyFilter()
{
    applySepiaFilter();
}

void GraphicsProcessing::loadImage(const QString& fileName)
{
    QImageReader reader(fileName);
    reader.setAutoTransform(true);
    const QImage newImage = reader.read();
    if (newImage.isNull()) {
        QMessageBox::information(this, tr("Error"), tr("Could not load image."));
        return;
    }

    m_image = newImage;
    QLabel* imageLabel = findChild<QLabel*>();
    imageLabel->setPixmap(QPixmap::fromImage(m_image));
}

void GraphicsProcessing::applySepiaFilter()
{
    for (int y = 0; y < m_image.height(); ++y) {
        for (int x = 0; x < m_image.width(); ++x) {
            QRgb pixel = m_image.pixel(x, y);
            int gray = qGray(pixel);
            int sepiaR = qBound(0, gray + 100, 255);
            int sepiaG = qBound(0, gray + 50, 255);
            int sepiaB = qBound(0, gray, 255);
            m_image.setPixel(x, y, qRgb(sepiaR, sepiaG, sepiaB));
        }
    }

    QLabel* imageLabel = findChild<QLabel*>();
    imageLabel->setPixmap(QPixmap::fromImage(m_image));
}

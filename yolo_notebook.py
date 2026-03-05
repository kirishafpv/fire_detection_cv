import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from ultralytics import YOLO

    return (YOLO,)


@app.cell
def _(YOLO):
    model = YOLO("yolov8n.pt")
    model
    return (model,)


@app.cell
def _(model):
    model.train(
        data="dataset_yolo/data.yaml",
        epochs=50,
        imgsz=640
    )
    return


if __name__ == "__main__":
    app.run()

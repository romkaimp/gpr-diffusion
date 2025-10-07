import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Важно для 3D проекции
import mlflow
import io
import mlflow
from PIL import Image

def lorenz_visualize(trajectory, prediction, name=None, file=None, true=None, ml_flow=False):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    print(trajectory.shape, prediction.shape)
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            lw=0.3, color='blue', alpha=0.8)
    ax.plot(prediction[:, 0], prediction[:, 1], prediction[:, 2],
            lw=0.4, color='orange', alpha=0.9)
    if true is not None:
        print("true:", true.shape)
        ax.plot(true[:, 0], true[:, 1], true[:, 2],
                lw=0.3, color='green', alpha=0.8)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    if name is None:
        name = "Predicted data"
    if file is None:
        file = name
    ax.set_title(name)
    if ml_flow:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        image = Image.open(buf)
        mlflow.log_image(image, f"lorenz_plots/{file}.png")

        image.close()
    else:
        plt.savefig(f'{file}.png', dpi=300, bbox_inches='tight')
        plt.show()
    plt.close(fig)
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Tuple


def load_dataset(path: str) -> Tuple[List[str], List[List[float]], List[float]]:
    """Load the EU COB volume data from the provided Excel file."""
    with zipfile.ZipFile(path) as z:
        shared = [
            ''.join(t.text for t in si.findall(
                './/{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t'
            ))
            for si in ET.parse(z.open('xl/sharedStrings.xml'))
            .getroot()
            .findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si')
        ]
        sheet = ET.parse(z.open('xl/worksheets/sheet1.xml')).getroot()
        rows = []
        for row in sheet.findall(
            './/{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row'
        ):
            vals = []
            for c in row.findall(
                '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'
            ):
                v = c.find(
                    '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v'
                )
                if v is not None:
                    if c.attrib.get('t') == 's':
                        vals.append(shared[int(v.text)])
                    else:
                        vals.append(v.text)
            rows.append(vals)

    header = rows[0]
    data = [list(map(float, r)) for r in rows[1:] if len(r) == len(header)]
    x_data = [[r[0]] + r[2:] for r in data]  # features (Time + others)
    y_data = [r[1] for r in data]  # TEU target
    return header, x_data, y_data


def normalise(data: List[List[float]]) -> Tuple[List[List[float]], List[float]]:
    """Scale columns to the [0, 1] range."""
    cols = list(zip(*data))
    max_vals = [max(col) for col in cols]
    scaled = [[row[i] / max_vals[i] for i in range(len(row))] for row in data]
    return scaled, max_vals


def scale_vector(vec: List[float]) -> Tuple[List[float], float]:
    m = max(vec)
    return [v / m for v in vec], m


def gradient_descent(
    X: List[List[float]],
    y: List[float],
    lr: float = 0.1,
    epochs: int = 2000,
) -> Tuple[List[float], float]:
    """Simple linear regression via gradient descent."""
    n = len(X)
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    for _ in range(epochs):
        dw = [0.0] * d
        db = 0.0
        for xi, yi in zip(X, y):
            pred = sum(w[j] * xi[j] for j in range(d)) + b
            err = pred - yi
            for j in range(d):
                dw[j] += err * xi[j]
            db += err
        for j in range(d):
            w[j] -= lr * dw[j] / n
        b -= lr * db / n
    return w, b


def main() -> None:
    header, x_raw, y_raw = load_dataset('dataset/EU COB volume dataset.xlsx')
    X, x_max = normalise(x_raw)
    y, y_max = scale_vector(y_raw)
    w, b = gradient_descent(X, y)

    print('Trained bias:', b)
    print('First three weights:', w[:3])

    # Display predictions for the first few samples
    for i in range(3):
        pred = sum(w[j] * X[i][j] for j in range(len(w))) + b
        print(
            f"Sample {i}: predicted={pred * y_max:.1f}, actual={y[i] * y_max:.1f}"
        )


if __name__ == '__main__':
    main()

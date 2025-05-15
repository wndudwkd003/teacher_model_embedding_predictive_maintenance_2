import pandas as pd, numpy as np, json, time

rows, cols = 100_000, 20
df = pd.DataFrame(np.random.rand(rows, cols), columns=[f'c{i}' for i in range(cols)])

t0 = time.perf_counter()
json_series = df.apply(lambda row: json.dumps(row.to_dict()), axis=1)
t_apply = time.perf_counter() - t0

t1 = time.perf_counter()
json_list = pd.DataFrame([json.dumps(rec) for rec in df.to_dict(orient='records')], columns=['json'])
t_orient = time.perf_counter() - t1

print(f"apply-lambda method: {t_apply:.3f} sec")
print(f"to_dict orient='records' method: {t_orient:.3f} sec")

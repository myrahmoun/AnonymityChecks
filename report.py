import itertools
import pandas as pd
import pycountry_convert as pc

K = 5
df = pd.read_csv("data.csv", dtype=str, keep_default_na=False).replace("null", "")
QI = [c for c in df.columns if c not in ("user_id", "course_id")]
FORUM = [c for c in QI if c.startswith("nforum_")]


def sizes(data, cols):
    """For each row, the number of rows that have the same values in these columns."""
    return data.groupby(cols)[cols[0]].transform("size")


# 1. Level of k-anonymity
s = sizes(df, QI)
print("1. k =", s.min())

# 2. Record suppression: delete every row whose group is smaller than K
print("2. records to delete:", (s < K).sum())

# 3. Column suppression: find the largest set of columns we can keep and still have k >= K.
# A subset of a valid set is also valid, so try keeping 1 column, then 2, ... until none work.
best = []
for n in range(1, len(QI) + 1):
    valid = [list(c) for c in itertools.combinations(QI, n) if sizes(df, list(c)).min() >= K]
    if not valid:
        break
    best = valid
print(f"3. columns to delete: {len(QI) - len(best[0])} of {len(QI)}")
for kept in best:
    print("   keep only:", kept, "-> delete:", [c for c in QI if c not in kept])

# 4. Generalization only: country -> continent, education kept, every other column
# generalized all the way to "*"
continent = {c: pc.country_alpha2_to_continent_code(c) for c in set(df.cc_by_ip) - {"", "TL"}}
continent.update({"": "", "TL": "AS"})  # TL (Timor-Leste) is missing from the library
g = df.copy()
g[["city", "postalCode", "YoB", "gender"] + FORUM] = "*"
g["cc_by_ip"] = g.cc_by_ip.map(continent)
print("4. k after generalization:", sizes(g, QI).min())

# 5. Combination: generalize less, suppress some columns, and delete a few records
g = df.copy()
g[["city", "postalCode"] + FORUM] = "*"                        # column suppression
g["cc_by_ip"] = g.cc_by_ip.map(continent)                      # generalization: continent
year = pd.to_numeric(g.YoB, errors="coerce")
year = year.where(year.between(1900, 2018))                   # values like 513 are errors
g["YoB"] = (year // 10 * 10).astype("Int64").astype("string").fillna("")  # generalization: decade
small = sizes(g, QI) < K
print(f"5. records to delete: {small.sum()} ({small.mean():.1%})")
print("   k afterwards:", sizes(g[~small], QI).min())

out = g[~small]
out.to_csv("data_5anon.csv", index=False)

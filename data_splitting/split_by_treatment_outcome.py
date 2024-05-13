from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go


def transform_treatment_info(x):
    if x["adjuvant_radiochemotherapy"] == "yes":
        return "radiochemotherapy"
    elif x["adjuvant_radiotherapy"] == "yes":
        return "radiotherapy"
    elif x["adjuvant_systemic_therapy"] == "yes":
        return "systemic therapy"
    else:
        return "none"


def transform_event_info(x):
    if x["survival_status"] == "deceased":
        return "yes"
    elif "yes" in x[["recurrence", "metastasis_1_locations", "progress_1"]].tolist():
        return "yes"
    else:
        return "no"


def define_dataset(x):
    if x["adjuvant_treatment"] == "none" and x["recurrent event or death"] == 1:
        return "test"
    else:
        return "training"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dir_structureddata", type=str, help="Path to directory containing structured data")
    parser.add_argument("dir_results", type=str, help="Path to directory for saving results")
    args = parser.parse_args()

    json_dir = Path(args.dir_structureddata)
    results_dir = Path(args.dir_results)

    df_orig = pd.read_json(json_dir/"clinical_data.json", dtype={"patient_id": str})
    df_orig = df_orig.merge(pd.read_json(json_dir/"pathological_data.json", dtype={"patient_id": str}), on="patient_id", how="inner")
    df = df_orig[[
        "patient_id",
        "first_treatment_modality",
        "adjuvant_radiotherapy",
        "adjuvant_systemic_therapy",
        "adjuvant_radiochemotherapy",
        "recurrence",
        "metastasis_1_locations",
        "progress_1",
        "survival_status"
    ]]
    df = df.fillna("no")
    df.metastasis_1_locations = df.metastasis_1_locations.apply(lambda x: "no" if x == "no" else "yes")
    df.first_treatment_modality = df.first_treatment_modality.replace("local surgery", "ablative surgery")

    # fig = px.parallel_categories(df)
    # fig.update_layout(width=1500, height=400)
    # fig.write_image(results_dir/"treatment_outcome_visualization_1.svg")

    # Summarize adjuvant treatments in one column
    df["adjuvant_treatment"] = df.apply(transform_treatment_info, axis=1)

    # Get recurrence-free survival
    df["recurrent event or death"] = df.apply(transform_event_info, axis=1)
    df = df[["patient_id", "first_treatment_modality", "adjuvant_treatment", "recurrent event or death"]]

    # fig = px.parallel_categories(df)
    # fig.update_layout(width=600, height=400)
    # fig.write_image(results_dir/"treatment_outcome_visualization_2.svg")

    # Visualization using plotly parallel categories
    df["recurrent event or death"] = df["recurrent event or death"].replace({"no": 0, "yes": 1})
    df["adj"] = df["adjuvant_treatment"].replace(
        {"none": 0, "radiotherapy": 1, "radiochemotherapy": 1, "systemic therapy": 1})
    df["no_adj_and_event"] = df.apply(
        lambda x: 0 if x["adjuvant_treatment"] == "none" and (x["recurrent event or death"] == 1) else 1, axis=1)

    # Create dimensions
    treatment_dim = go.parcats.Dimension(values=df.first_treatment_modality, label="first treatment")
    adjuvant_dim = go.parcats.Dimension(
        values=df.adjuvant_treatment, categoryorder="array", label="adjuvant treatment",
        categoryarray=["none", "radiotherapy", "systemic therapy", "radiochemotherapy"]
    )
    event_dim = go.parcats.Dimension(
        values=df["recurrent event or death"], label="recurrent event or death", categoryarray=[1, 0], ticktext=["yes", "no"]
    )
    # Create parcats trace
    color = df.no_adj_and_event
    colorscale = [[0, "sandybrown"], [1, "lightsteelblue"]]

    fig = go.Figure(data=[go.Parcats(dimensions=[treatment_dim, adjuvant_dim, event_dim],
                                     line={"color": color, "colorscale": colorscale, "shape": "hspline"},
                                     hoveron="color", hoverinfo="count+probability",
                                     arrangement="freeform")])
    fig.update_layout(width=600, height=400)
    fig.write_image(results_dir/"treatment_outcome_visualization.svg")

    # Save dataset split
    data_split = df[["patient_id", "adjuvant_treatment", "recurrent event or death"]].copy()
    data_split["dataset"] = data_split.apply(define_dataset, axis=1)
    data_split.to_json(results_dir/"dataset_split_treatment_outcome.json", orient="records", indent=1)

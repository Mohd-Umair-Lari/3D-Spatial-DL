# Aventix Scarlett Real-Data KGRAG

This project contains a Python/Jupyter implementation of a grounded Knowledge Graph Retrieval-Augmented Generation (KGRAG) pipeline for real SCARLETT optical-scanner point-cloud data from the Aventix Combi dispense experiment.

The primary artifact is `aventix_kgrag_real.ipynb`. It uses only the real CSV dataset under:

```text
dataset/Point_cloud_dataset/Point cloud dataset - Combi dispense 2026-11/
├── manifest.csv
├── wells.csv
├── scans/
└── empty_reads/
```

The source data is read-only; the notebook does not modify the original scans.

## Objective

The notebook converts filled and empty point-cloud scans into traceable per-well measurements, deterministic verification outcomes, and an in-memory NetworkX knowledge graph. An optional Groq LLM explains retrieved graph facts, but it does not decide measurements, statuses, or faults.

The dataset has no per-well gravimetric ground truth. Gravimetry is therefore used only as a plate-level reference; per-well values are scanner-derived estimates.

## Dataset

The real dataset contains 15 filled scans across eight plates:

- EP384-A: water, cumulative steps
- AB0796-B: water, cumulative steps
- AB0796-C: water, ladder fill
- AB0796-D: DMSO, ladder fill
- AB0796-E: DMSO, cumulative steps
- AB0796-F: DMSO, partial fill
- AB0796-G: DMSO, single fill
- AB0796-H: DMSO, single fill

`manifest.csv` contains scan metadata, liquid, fill step, empty-read mapping, mass measurements, density, and plate-level gravimetric mean. `wells.csv` contains per-scan well centers and nominal volumes. Point files contain `well`, `row`, `col`, `x_mm`, `y_mm`, `z_mm`, and `sensor`.

## Notebook pipeline

`aventix_kgrag_real.ipynb` performs the following:

1. Locates and audits the real dataset.
2. Loads the manifest, well centers, filled scans, and empty reads.
3. Isolates each well using its measured center and plate-format radius.
4. Fuses both optical sensors and removes Z outliers.
5. Estimates the empty-well floor and subtracts it from the filled scan.
6. Applies simple surface-tension and meniscus corrections using liquid, density, and radius.
7. Produces raw and calibrated per-well volume estimates.
8. Compares `estimated_uL` with per-well `nominal_uL`.
9. Applies deterministic verification rules.
10. Builds a NetworkX knowledge graph with measurement provenance.
11. Answers grounded questions using graph facts and optional Groq synthesis.

## Measurement columns

- `nominal_uL`: Expected volume for the individual well from `wells.csv`.
- `raw_geometric_uL`: Cylindrical volume from the uncorrected Z height.
- `capillary_height_mm`: Estimated capillary rise from surface tension.
- `meniscus_correction_uL`: Approximate meniscus adjustment.
- `estimated_uL`: Corrected and scan-calibrated per-well estimate.
- `gravimetric_mean_uL_per_well`: Plate-level balance-derived reference.
- `deviation_pct`: Difference between `estimated_uL` and `nominal_uL`.
- `gravimetric_deviation_pct`: Difference between `estimated_uL` and the plate-level gravimetric mean.
- `baseline_estimated`: Whether a substitute empty-read baseline was required.
- `status`, `fault_reason`, `rule_id`: Deterministic verification output.

The notebook currently writes `measurments.csv` for backward compatibility; this filename can later be corrected to `measurements.csv`.

## Knowledge graph and RAG

The graph is built in memory with NetworkX and represents Lab, Run, Plate, Well, Measurement, Instrument, plate format, liquid, fill steps, and empty reads. Every measurement retains its source scan and verification details.

The LLM receives only retrieved graph facts. If Groq is unavailable, deterministic graph-grounded responses remain available.

## Running the notebook

```bash
pip install -r requirements.txt
jupyter notebook aventix_kgrag_real.ipynb
```

Run cells from top to bottom. The notebook expects the real dataset at the relative path shown above.

For optional Groq responses, create `.env` in the project root:

```text
GROQ_API_KEY=your_groq_api_key
MODEL=llama-3.3-70b-versatile
```

Without `GROQ_API_KEY`, deterministic processing and graph retrieval still work.

## Data safety

The notebook reads the real CSV scans and writes only derived analysis output. It does not overwrite, rename, or regenerate the original point-cloud data.

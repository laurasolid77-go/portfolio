# Scope

These rules apply only to the `molecular-screening-webapp` project folder.

Do not apply these rules to other projects in the same workspace.

Do not read, edit, move, rename, or delete files outside the `molecular-screening-webapp` folder unless explicitly instructed by the user.

# Project Rules

This project is a portfolio-level Streamlit web app for molecular screening using a PubChem-derived molecular library and RDKit.

## Core Goal

Build an interactive screening-based web app that loads a pre-built PubChem-derived molecular library from a local CSV file and allows users to screen molecules based on RDKit-derived structural descriptors.

The main Streamlit app is not a live PubChem keyword search app. PubChem data should be collected in a separate data preparation step, and the app should focus on screening the prepared molecular library.

## Scope

- Keep the app simple, stable, and readable.
- Use Streamlit only for the web app.
- Use PubChemPy only for offline molecular library construction, not for live search inside the main Streamlit app.
- Use RDKit for molecular descriptor calculation and 2D structure visualization.
- Use pandas for data handling.
- Use plotly for visualization.
- Do not add deep learning, GNN, docking, quantum chemistry, FastAPI, Flask, React, or databases unless explicitly requested.
- Do not attempt to download the entire PubChem database.
- The app should load a local CSV file such as `data/processed/molecular_library.csv`.

## Sequential Thinking and Execution

- Work through the project step by step.
- Before making changes, briefly explain the next step and why it is needed.
- Break large tasks into small, testable implementation steps.
- Do not implement multiple major features at once.
- After each feature is implemented, explain how to test it.
- Verify that the current step works before moving to the next step.
- If an error occurs, diagnose the cause first before modifying unrelated code.
- Prefer incremental progress over large refactors.
- Keep track of what has been completed and what remains to be done.
- When uncertain, choose the simpler and more stable implementation.

## Scientific Claims

- Do not claim prediction of HOMO/LUMO, redox potential, solubility, electrochemical stability, or real material performance.
- The screening score must always be described as a heuristic structure-based first-screening index.
- State that DFT calculations and experimental validation are required for real material evaluation.
- Use cautious language such as:
  - first-screening
  - candidate prioritization
  - structure-based filtering
  - descriptor-based screening
  - heuristic index

## Required Features

- Load a pre-built PubChem-derived molecular library from a local CSV file.
- Screen molecules using RDKit-derived structural descriptors.
- RDKit descriptor calculation or descriptor validation when needed.
- Molecular weight filter.
- Carbonyl count filter.
- Aromatic ring count filter.
- Atom count filters.
- Heteroatom and halogen filters.
- Filtered result table.
- Top candidate 2D structure images.
- Descriptor plots.
- CSV download.

## Data Preparation Rules

- PubChemPy may be used in a separate script or notebook to build the molecular library.
- The main Streamlit app should not perform live PubChem keyword search.
- The data preparation step may retrieve:
  - CID
  - IUPAC name
  - molecular formula
  - molecular weight
  - canonical SMILES
  - isomeric SMILES
  - XLogP
  - TPSA
  - charge
- Drop duplicate CIDs.
- Remove rows without canonical SMILES.
- Handle empty PubChem results and API errors gracefully in the data preparation step.
- Save the final processed library as a local CSV file under `data/processed/`.

## RDKit Descriptors

Calculate or validate the following descriptors when possible:

- molecular weight
- exact molecular weight
- LogP
- TPSA
- H-bond donor count
- H-bond acceptor count
- rotatable bond count
- ring count
- aromatic ring count
- heavy atom count
- fraction CSP3
- carbonyl count using SMARTS pattern `[CX3]=[OX1]`
- C atom count
- O atom count
- N atom count
- S atom count
- B atom count
- F atom count
- Cl atom count
- Br atom count
- I atom count
- halogen count
- heteroatom count

Invalid SMILES must be handled safely.

## Screening Score

Use this heuristic score unless explicitly changed:

```text
screening_score =
2.0 * carbonyl_count
+ 1.0 * aromatic_ring_count
+ 0.5 * heteroatom_count
- 0.01 * molecular_weight
- 0.2 * rotatable_bonds
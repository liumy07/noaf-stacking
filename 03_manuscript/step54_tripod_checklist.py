# -*- coding: utf-8 -*-
"""
step54_tripod_checklist.py — build the TRIPOD+AI submission checklist (docx).

Items and wording follow the official checklist, Version 11-January-2024:
Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance
for reporting clinical prediction models that use regression or machine
learning methods. BMJ 2024;385:e078378. doi:10.1136/bmj-2023-078378
https://www.tripod-statement.org

The "Reported in manuscript" column maps every item to the section of
NOAF_stacking_manuscript_EndNote_revised.docx where it is addressed.
"""
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

OUT = r"<LOCAL_ROOT>/WorkBuddy/2026-09-09-17-02-31/output/rev_noaf/TRIPOD+AI_checklist_NOAF_stacking.docx"

# (section, item, D/E, checklist item, reported in manuscript)
ROWS = [
 ("TITLE", "1", "D;E",
  "Identify the study as developing or evaluating the performance of a multivariable prediction model, the target population, and the outcome to be predicted",
  "Title page: development and external validation of a new-onset atrial fibrillation prediction model for critically ill adults"),
 ("ABSTRACT", "2", "D;E",
  "See TRIPOD+AI for Abstracts checklist",
  "Abstract (structured: background, methods, results incl. participant numbers, performance with 95% CI, interpretation)"),
 ("INTRODUCTION", "3a", "D;E",
  "Explain the healthcare context (including whether diagnostic or prognostic) and rationale for developing or evaluating the prediction model, including references to existing models",
  "Introduction, paragraphs 1-3 (prognostic context; prior NOAF prediction models cited and their limitations)"),
 ("", "3b", "D;E",
  "Describe the target population and the intended purpose of the prediction model in the context of the care pathway, including its intended users",
  "Introduction, final paragraph (critically ill adults in ICUs; early risk stratification to guide surveillance; intended for ICU clinicians)"),
 ("", "3c", "D;E",
  "Describe any known health inequalities between sociodemographic groups",
  "Discussion, limitations paragraph (two US databases; generalizability to other health-care systems discussed)"),
 ("", "4", "D;E",
  "Specify the study objectives, including whether the study describes the development or validation of a prediction model (or both)",
  "Introduction, final sentence (development with internal validation and external validation in eICU-CRD)"),
 ("METHODS / Data", "5a", "D;E",
  "Describe the sources of data separately for the development and evaluation datasets, the rationale for using these data, and representativeness of the data",
  "Section 2.1 (MIMIC-IV v3.1 for development; eICU-CRD v2.0 for external evaluation; both de-identified public critical-care databases)"),
 ("", "5b", "D;E",
  "Specify the dates of the collected participant data, including start and end of participant accrual",
  "NOT YET STATED in Section 2.1 — recommend adding accrual dates (MIMIC-IV 2008-2022; eICU-CRD 2014-2015) before submission"),
 ("Participants", "6a", "D;E",
  "Specify key elements of the study setting including the number and location of centres",
  "Section 2.1-2.2 (Beth Israel Deaconess Medical Center [MIMIC-IV]; >200 US ICUs [eICU-CRD])"),
 ("", "6b", "D;E",
  "Describe the eligibility criteria for study participants",
  "Section 2.2 (age >18 y; ICU stay >2 days; no AF on ICU day 1; no history of AF; no prior cardiac procedures including valve operations and CABG; first ICU admission only)"),
 ("", "6c", "D;E",
  "Give details of any treatments received, and how they were handled, if relevant",
  "Sections 2.3 and Table 1 (vasopressor and antibiotic use included as candidate predictors)"),
 ("Data preparation", "7", "D;E",
  "Describe any data pre-processing and quality checking, including whether this was similar across relevant sociodemographic groups",
  "Section 2.3 (unit harmonisation across the two databases, range and consistency checks; identical pipeline applied to both cohorts)"),
 ("Outcome", "8a", "D;E",
  "Clearly define the outcome being predicted and the time horizon, including how and when assessed, and the rationale",
  "Section 2.3 (new-onset atrial fibrillation documented after the first 24 h of ICU admission in patients without pre-existing AF, following the deferred-outcome-window convention; identified from the structured heart-rhythm record of the ICU stay)"),
 ("", "8b", "D;E",
  "If outcome assessment requires subjective interpretation, describe the qualifications of the outcome assessors",
  "Section 2.3 (clinician-documented diagnoses; ICD/coding as recorded by treating teams)"),
 ("", "8c", "D;E",
  "Report any actions to blind assessment of the outcome to be predicted",
  "Section 2.3 (outcome defined before modelling; predictors recorded in the first 24 h, preceding the outcome)"),
 ("Predictors", "9a", "D",
  "Describe the choice of initial predictors and any pre-selection of predictors before model building",
  "Sections 2.3-2.4 (39 candidate predictors selected from prior literature and data availability; univariable screen at P<0.05, then LASSO 1-SE rule)"),
 ("", "9b", "D;E",
  "Clearly define all predictors, including how and when they were measured",
  "Section 2.3 (demographics, comorbidities, first-24-h vital signs and laboratory extremes, treatments; identical definitions harmonised across databases)"),
 ("", "9c", "D;E",
  "If predictor measurement requires subjective interpretation, describe the qualifications of the predictor assessors",
  "Not applicable (all predictors are objective structured charting variables)"),
 ("Sample size", "10", "D;E",
  "Explain how the study size was arrived at and justify that the study size was sufficient, separately for development and evaluation",
  "Sections 2.2 and 3.1 (34,384 MIMIC-IV patients with 9,535 NOAF events [27.7%] for development; 32,067 eICU-CRD patients with 2,457 events [7.7%] for evaluation)"),
 ("Missing data", "11", "D;E",
  "Describe how missing data were handled. Provide reasons for omitting any data",
  "Section 2.3 (variables >30% missing excluded [8 laboratory variables]; missForest-type random-forest chained-equation imputation fitted on the 70% training rows only; 26/39 predictors had missing values in MIMIC-IV, median 1.2%, maximum 6.5%; variable-wise missingness in Supplementary Table S8; fold-internal refitting sensitivity analysis in Supplementary Table S6)"),
 ("Analytical methods", "12a", "D",
  "Describe how the data were used, including whether the data were partitioned, considering sample size requirements",
  "Sections 2.4 (70/30 stratified split, seed fixed; 10-fold CV for out-of-fold predictions)"),
 ("", "12b", "D",
  "Describe how predictors were handled in the analyses (functional form, rescaling, transformation, standardisation)",
  "Section 2.5 (standardisation of continuous predictors; one-hot encoding of categorical predictors inside every pipeline)"),
 ("", "12c", "D",
  "Specify the type of model, rationale, all model-building steps, including any hyperparameter tuning, and method for internal validation",
  "Sections 2.4-2.6 (univariable screen + LASSO 1-SE re-run inside every CV fold; nine learners with fixed hyperparameters; LASSO meta-learner; 10-fold CV out-of-fold predictions)"),
 ("", "12d", "D;E",
  "Describe if and how any heterogeneity in estimates of model parameter values and model performance was handled and quantified across clusters",
  "Section 3.4 and Discussion (external cohort is a distinct multi-centre cluster; heterogeneity quantified by internal-vs-external performance comparison; per-centre analysis not performed - see Limitations)"),
 ("", "12e", "D;E",
  "Specify all measures and plots used to evaluate model performance and to compare multiple models",
  "Section 2.7 (AUC, average precision, Brier score, calibration intercept/slope, calibration curves, decision curves, net benefit; DeLong test for paired AUC comparisons)"),
 ("", "12f", "E",
  "Describe any model updating (e.g., recalibration) arising from the model evaluation",
  "Not performed; discussed as a next step in the Discussion (recalibration in external settings)"),
 ("", "12g", "E",
  "For model evaluation, describe how the model predictions were calculated",
  "Section 2.6 (one frozen predictor: locked variable set + nine refitted learners + LASSO meta-learner; eICU predictions from the identical object)"),
 ("Class imbalance", "13", "D;E",
  "If class imbalance methods were used, state why and how this was done, and any subsequent methods to recalibrate the model",
  "Section 2.5 (no resampling or class weighting applied; models trained at the observed prevalence of approximately 28%)"),
 ("Fairness", "14", "D;E",
  "Describe any approaches that were used to address model fairness and their rationale",
  "Not formally assessed; acknowledged in the Limitations (fairness evaluation across demographic subgroups as future work)"),
 ("Model output", "15", "D",
  "Specify the output of the prediction model (e.g., probabilities, classification) and how thresholds were identified",
  "Section 2.6 (predicted probability of NOAF; no fixed classification threshold imposed - decision curves shown across the clinically plausible threshold range)"),
 ("Training versus evaluation", "16", "D;E",
  "Identify any differences between the development and evaluation data in healthcare setting, eligibility criteria, outcome, and predictors",
  "Sections 2.1-2.3 and Table 1 (setting, era and case-mix differences; standardized mean differences reported for all 39 predictors)"),
 ("Ethical approval", "17", "D;E",
  "Name the institutional research board or ethics committee and describe informed consent or the waiver",
  "Declarations, Ethics approval and consent to participate (de-identified public databases under PhysioNet data-use agreements; first author CITI-certified, No. 73146173; the Ethics Committee of the First Affiliated Hospital of Dalian Medical University waived ethical approval)"),
 ("OPEN SCIENCE / Funding", "18a", "D;E",
  "Give the source of funding and the role of the funders",
  "Declarations, Funding (this research received no specific grant from any funding agency)"),
 ("", "18b", "D;E",
  "Declare any conflicts of interest and financial disclosures for all authors",
  "Declarations, Conflicts of interest (the authors declare no competing interests)"),
 ("", "18c", "D;E",
  "Indicate where the study protocol can be accessed or state that a protocol was not prepared",
  "Declarations (no separate study protocol was prepared)"),
 ("", "18d", "D;E",
  "Provide registration information or state that the study was not registered",
  "Declarations (not registered; retrospective analysis of de-identified public databases)"),
 ("", "18e", "D;E",
  "Provide details of the availability of the study data",
  "Declarations, Data availability (MIMIC-IV v2.2 and eICU-CRD v2.0 available to credentialed users via PhysioNet)"),
 ("", "18f", "D;E",
  "Provide details of the availability of the analytical code",
  "Declarations, Code availability (complete analysis code NOAF_stacking_analysis_code.zip, deposited with the manuscript)"),
 ("Patient & Public Involvement", "19", "D;E",
  "Provide details of any patient and public involvement or state no involvement",
  "Declarations (no patient or public involvement)"),
 ("RESULTS / Participants", "20a", "D;E",
  "Describe the flow of participants through the study, including numbers with and without the outcome. A diagram may be helpful",
  "Section 3.1 and Figure 1 (participant flow diagram for both cohorts)"),
 ("", "20b", "D;E",
  "Report the characteristics overall and for each data source, including key dates, key predictors, treatments, sample size, outcome events, and amount of missing data",
  "Section 3.1 and Table 1 (baseline characteristics with standardized mean differences; missingness in Supplementary Table S1)"),
 ("", "20c", "E",
  "For model evaluation, show a comparison with the development data of the distribution of important predictors",
  "Table 1 (SMD column; differences discussed in Section 3.1)"),
 ("Model development", "21", "D;E",
  "Specify the number of participants and outcome events in each analysis (development, hyperparameter tuning, evaluation)",
  "Sections 3.1-3.3 (24,068 training / 10,316 internal test / 32,067 eICU; events per analysis stated)"),
 ("Model specification", "22", "D",
  "Provide details of the full prediction model (formula, code, object, API) to allow predictions in new individuals",
  "Code availability (complete reproducible pipeline, including the frozen predictor object and all hyperparameters)"),
 ("Model performance", "23a", "D;E",
  "Report model performance estimates with confidence intervals, including for any key subgroups. Consider plots",
  "Sections 3.3-3.4, Table 2, Figures 2-5 (AUC/AP with 95% CI, calibration, DCA; no subgroup estimates - noted in Limitations)"),
 ("", "23b", "D;E",
  "If examined, report results of any heterogeneity in model performance across clusters",
  "Not formally examined; internal-vs-external comparison in Section 3.4 addresses transportability; per-centre heterogeneity noted as future work"),
 ("Model updating", "24", "E",
  "Report the results from any model updating, including the updated model and subsequent performance",
  "Not performed (see item 12f)"),
 ("DISCUSSION / Interpretation", "25", "D;E",
  "Give an overall interpretation of the main results, including issues of fairness in the context of the objectives and previous studies",
  "Discussion, paragraphs 1-3 (interpretation in the context of prior NOAF models; comparison of discriminative advantage)"),
 ("", "26", "D;E",
  "Discuss any limitations of the study and their effects on biases, statistical uncertainty, and generalizability",
  "Discussion, Strengths and limitations paragraph (retrospective design, US-only databases, missing data, no external recalibration, fairness)"),
 ("Usability in current care", "27a", "D",
  "Describe how poor quality or unavailable input data should be assessed and handled when implementing the model",
  "Discussion (first-24-h data availability in routine ICU charting; missingness handled by the imputation strategy described in Section 2.3)"),
 ("", "27b", "D",
  "Specify whether users will be required to interact in the handling of input data or use of the model, and what expertise is required",
  "Discussion (electronic health record integration with automated extraction of the 26 selected variables; intended for use by ICU clinicians)"),
 ("", "27c", "D;E",
  "Discuss any next steps for future research, with a view to applicability and generalizability",
  "Discussion, final paragraph (prospective validation, recalibration in external settings, temporal and geographic transportability, decision-impact studies)"),
]


def main():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Arial"
    st.font.size = Pt(9)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")

    for s in doc.sections:
        s.left_margin = Cm(1.8)
        s.right_margin = Cm(1.8)
        s.top_margin = Cm(1.8)
        s.bottom_margin = Cm(1.8)

    p = doc.add_paragraph()
    r = p.add_run("Supplementary Table S9. TRIPOD+AI checklist — completed for the submitted manuscript")
    r.bold = True
    r.font.size = Pt(13)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p = doc.add_paragraph(
        "Manuscript: Prediction model for new-onset atrial fibrillation in critically ill "
        "patients using a stacking ensemble strategy: development and external validation "
        "based on MIMIC-IV and eICU-CRD.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p = doc.add_paragraph(
        "Checklist items follow the official TRIPOD+AI statement, Version 11-January-2024: "
        "Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for "
        "reporting clinical prediction models that use regression or machine learning methods. "
        "BMJ 2024;385:e078378. doi:10.1136/bmj-2023-078378 (www.tripod-statement.org). "
        "D = development-only item; E = evaluation-only item; D;E = relevant to both.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs:
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0x40, 0x40, 0x40)

    t = doc.add_table(rows=1, cols=5)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    widths = [Cm(2.6), Cm(1.1), Cm(1.1), Cm(8.2), Cm(4.8)]
    hdr = ["Section/Topic", "Item", "D/E", "Checklist item", "Reported in manuscript"]
    for j, (h, w) in enumerate(zip(hdr, widths)):
        c = t.rows[0].cells[j]
        c.width = w
        pr = c.paragraphs[0]
        run = pr.add_run(h)
        run.bold = True
        run.font.size = Pt(9)

    for sec, item, de, text, rep in ROWS:
        cells = t.add_row().cells
        for j, w in enumerate(widths):
            cells[j].width = w
        vals = [sec, item, de, text, rep]
        for j, v in enumerate(vals):
            pr = cells[j].paragraphs[0]
            run = pr.add_run(v)
            run.font.size = Pt(8.5)
            if j == 0 and v:
                run.bold = True

    p = doc.add_paragraph()
    r = p.add_run("Page numbers should be inserted in the final column after the "
                  "manuscript is paginated for submission.")
    r.font.size = Pt(8.5)
    r.italic = True
    r.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

    doc.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()

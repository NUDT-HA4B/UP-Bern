# BC4CHEMD Dataset

## Overview and Dataset Details

The **BC4CHEMD** dataset contains **10,000** PubMed abstracts annotated with **chemical compounds** and **drug names**. The dataset is split as follows:
- **6,000** abstracts for training
- **2,000** abstracts for validation
- **2,000** abstracts for testing

The dataset includes the following components:
1. **Plain Text Abstracts**: Texts for the training, validation, test, and background sets.
2. **Manual Gold Standard Annotations**: High-quality expert annotations.
3. **Task Team Predictions**: Predictions made by the task team.
4. **Silver Standard Corpus**: Automatically generated annotations with lower precision.
5. **Chemical Disciplines Subsets**: Subsets focused on specific chemical disciplines.
6. **BioC Version of Annotations**: Annotations in BioC format for compatibility with NLP tools.

**Entity Types**: Chemical Compounds, Drug Names. This dataset serves as a benchmark for chemical entity recognition and normalization tasks in biomedical text processing.

## Usage

This dataset is primarily used for:
- **Chemical Entity Recognition**: Identifying chemical compounds and drug names in abstracts.
- **Entity Normalization**: Linking identified entities to their corresponding identifiers.

## Citation

If you use this dataset in your research, please cite the following paper:

- Krallinger, M., et al. **The CHEMDNER Corpus of Chemicals and Drugs and Its Annotation Principles**. *J Cheminform*, 2014.

## License

This dataset is available under the BioCreative corpus license. Please refer to the official BioCreative website for more details on usage rights and citation.

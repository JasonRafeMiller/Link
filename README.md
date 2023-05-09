# Link

## Table 1. 
Middle-exclusion, lncRNA, traditional machine learning.   
* LRF_101: Random forest no exclusion.   
* LRF_102: Random forest with middle exclusion.   
* LGB_101: Gradient boosting no exclusion.   
* LGB_102: Gradient boosting with middle exclusion.   
* LSVM_101: Support vector machine no exclusion.
* LSVM_102: Support vector mcahine with middle exclusion.

## Table 2.
Middle-exclusion, lncRNA, neural network. 
* MLP_NoNo: Regime A.
* MLP_YesYes: Regime B.
* MLP_YesNo: Regime C.
* MLP_NoYes3: Regime D.
* Stats1: statistical significance

## Table 3. 
Middle-exclusion, mRNA, neural network.
* MLP_mRNA_NoNo: Regime A. 
* MLP_mRNA_YesYes: Regime B.
* MLP_mRNA_YesNo: Regime C.
* MLP_mRNA_NoYes: Regime D.
* Stats1: statistical significance

## Table 4. 
Cross-validation and test results.
* MLP_NoNo: lncRNA cross-valiation. 
* MLP_NoNo_Test: lncRNA test results.
* MLP_mRNA_NoNo: mRNA cross-validation.
* MLP_mRNA_Test. mRNA test results.

## Table 5. 
Canonical vs longest and all lncRNA transcripts per gene.
* MLP_NoNo: lncRNA canonical transcript.
* MLP_longest: lncRNA longest transcript.
* MLP_all: lncRNA all transcripts.

## Table 6. 
Canonical vs longest and all mRNA transcripts per gene.
* MLP_mRNA_NoNo: mRNA canonical transcript.
* MLP_mRNA_longest: mRNA longest transcript.
* MLP_mRNA_all: mRNA all transcripts.

## Table 7.
MLP with GMM instead of thrshold on lncRNA.
* GMM_CV: Mean of 14 cell lines, cross-validation.
* GMM_Test: Mean of 14 cell lines, test results.
* CellLine_01: Cell line H1.hESC, cross-validation.
* CellLine_01_Test: Cell line H1.hESC, test results.

## Figure S1.
Training histories, learning failures.
* MLP_mRNA_NoNo: Top left, success with mean mRNA.
* MLP_NoNo: Top right, success with mean lncRNA.
* CellLine_03: Bottom left, success with single-cell line lncRNA.
* CellLine_03: Bottom right, failure with single-cell line lncRNA.

## Table S3.
MLP cross-validation results on individual cell lines.  
* CellLine_00: A549
* CellLine_01: H1.hESC
* CellLine_02: HeLa.S3
* CellLine_03: HepG2
* CellLine_04: HT1080
* CellLine_05: HUVEC
* CellLine_06: MCF.7
* CellLine_07: NCI.H460
* CellLine_08: NHEK
* CellLine_09: SK.MEL.5
* CellLine_10: SK.N.DZ
* CellLine_11: SK.N.SH
* CellLine_12: GM12878
* CellLine_13: K562
* CellLine_14: IMR.90

## Table S4.
GMM latent parameters inferred on the 14-cell-line means. 
* GMM_CV

## Table S5.
GMM latent parameters inferred on cell line H1.hESC. 
* CellLine_01: H1.hESC

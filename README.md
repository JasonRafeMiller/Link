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

## Table 3. 
Middle-exclusion, mRNA, neural network.
* MLP_mRNA_NoNo: Regime A. 
* MLP_mRNA_YesYes: Regime B.
* MLP_mRNA_YesNo: Regime C.
* MLP_mRNA_NoYes: Regime D.

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

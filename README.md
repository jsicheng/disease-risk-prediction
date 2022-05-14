
Information about the genotype and phenotype information in this folder


These folders contain real genotype data from the European individuals in the 1000 Genomes data as well as continuous and case control traits that have been simulated using GCTA for different levels of heritablity and proportions of causal SNPs.


Genotype Data (./Genotypes/):

The chromosome 22 imputed data was filtered for MAF > 0.05, HWE p-value > 0.000005, only keeping variation that had a PASS filter  and for only containging SNPs. The genotype data is available in both the vcf format as well as in plink format with a bed, bim and fam file. The bim file contains information about the SNPs in your data, the fam file about the individuals and the bed file the plink-readable genotype matrix.


Phenotype Data:

Paramenters
1) 'hsq' refers to the proportion of the variance in the simulated trait that is due to genetics. Simulations use hsq values of 0.0, 0.2, 0.4, 0.6 and 0.8. Each phenotype filename contains the hsq value use to make the file.
2) 'rep' refers to repeates of each simulation. The rep number {1..5} comes before the .log, .par or .phen suffix on each filename. While each repeat is done the same way the effect sizes for the causal snps will be different.
3) There are two often used models for simulating complex phenotype times, the infinitesimal model (allcausal) that assumes every SNP affects the trait, and Non-infinitesimal (OnePercCausal) where some SNPs have no effect on the trait, we assume that one percent of SNPs have a non-zero effect.
4) Case-Control vs Continuous - Height is an example of a continous trait while has diabetes vs doesn't have diabetes is an example of a Case-Control trait. Continuous traits are simulated by assigning each SNP an affect size, then multiplying the SNP genotypes values by the effect size. Usually the genotypes values are first standardized to have mean zero and variance 1, this is done by GCTA in this simulation. For our Case-Control simulations, we we simulate a continuous trait, and then take the 50% of individuals with the highest values as cases and the 50% with the lowest as controls. This simulation would be modeling a binary trait where 50% of the population would be considered a case (ie prevalence of 50%), this is not a realistic simulation for the vast majority of case-control traits.



Files
1) par file - the par file gives information about the effect size of each SNP on the trait. Only SNPs with non-zero effect size are reported. Each simulation has different causal SNPs.
2) phen file - This contains the simulated phenotype information. The first column contains the family id, in this case it is all 0s and can be ignored. The second column contains the individual IDs that will allow you to link this file to the genotype file. The third column contains the simulated phenotype.
3) log file - you can read this file to see exactly how the simulation was performed that created the par and phen files.


All the file names for the phenotypes contain all the information about how the simulation was performed. They have been divided into four folders for clarity.

./Phenotypes/InfCaseCont - Infinitesimal model with case-control phenotypes
./Phenotypes/InfCont - Infinitesimal model with continuous phenotypes
./Phenotypes/OnePercCaseCont - Non-infinitesimal model with approximately one percent of SNPs being causal and case control phenotypes 
./Phenotypes/OnePercCont - Non-infinitesimal model with approximately one percent of SNPs being causal and continuous phenotypes 

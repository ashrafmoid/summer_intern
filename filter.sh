#!/bin/bash
cat gencode.v26.annotation.gff3 | awk '/\sexon\s|\stranscript\s|\sgene\s/ && /gene_type=protein_coding/' > gencode_protein_coding.gff

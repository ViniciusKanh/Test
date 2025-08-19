import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2, f
import warnings

class FriedmanTestFinal:
    """
    Implementação final do teste de Friedman usando scipy e metodologia do scmamp
    """
    
    def __init__(self, data):
        """
        Inicializa com os dados
        data: DataFrame com datasets nas linhas e algoritmos nas colunas
        """
        self.data = data.copy()
        self.n_datasets = len(data)
        self.n_algorithms = len(data.columns)
        self.algorithm_names = list(data.columns)
        
        # Calcular rankings
        self.ranks = self._calculate_ranks()
        self.mean_ranks = self.ranks.mean()
        
    def _calculate_ranks(self):
        """
        Calcula os rankings para cada dataset
        Ranking 1 = melhor performance, ranking k = pior performance
        """
        # Para cada linha (dataset), ranquear os algoritmos
        # rank(method='min', ascending=False) - maior valor recebe rank 1
        ranks = self.data.rank(axis=1, method='min', ascending=False)
        return ranks
    
    def friedman_test_scipy(self):
        """
        Usa a implementação do scipy para o teste de Friedman
        """
        # Preparar dados para scipy (cada coluna é um grupo)
        data_arrays = [self.data[col].values for col in self.data.columns]
        
        # Executar teste de Friedman do scipy
        statistic, p_value = stats.friedmanchisquare(*data_arrays)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'df': self.n_algorithms - 1,
            'test_name': 'Friedman Test (scipy)'
        }
    
    def iman_davenport_test(self):
        """
        Executa o teste de Friedman com correção de Iman-Davenport
        """
        N = self.n_datasets
        k = self.n_algorithms
        
        # Usar o teste de Friedman do scipy
        scipy_result = self.friedman_test_scipy()
        chi2_stat = scipy_result['statistic']
        
        # Correção de Iman-Davenport
        # F = ((N-1) * χ²) / (N(k-1) - χ²)
        numerator = (N - 1) * chi2_stat
        denominator = N * (k - 1) - chi2_stat
        
        if denominator > 0:
            f_stat = numerator / denominator
            df1 = k - 1
            df2 = (N - 1) * (k - 1)
            p_value = 1 - f.cdf(f_stat, df1, df2)
        else:
            # Fallback para chi-quadrado se denominador for problemático
            f_stat = chi2_stat / (k - 1)
            df1 = k - 1
            df2 = float('inf')
            p_value = scipy_result['p_value']
        
        return {
            'statistic': f_stat,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'df1': df1,
            'df2': df2,
            'test_name': 'Iman-Davenport correction of Friedman Test'
        }
    
    def _pairwise_comparisons(self):
        """
        Calcula comparações par-a-par entre algoritmos usando teste de Wilcoxon
        """
        from scipy.stats import wilcoxon
        
        comparisons = []
        raw_pvalues = {}
        
        for i, alg1 in enumerate(self.algorithm_names):
            raw_pvalues[alg1] = {}
            for j, alg2 in enumerate(self.algorithm_names):
                if i != j:
                    # Usar teste de Wilcoxon para comparação par-a-par
                    try:
                        # Diferenças entre os algoritmos
                        diff = self.data[alg1] - self.data[alg2]
                        # Remover zeros (empates)
                        diff_nonzero = diff[diff != 0]
                        
                        if len(diff_nonzero) > 0:
                            statistic, p_val = wilcoxon(diff_nonzero, alternative='two-sided')
                        else:
                            p_val = 1.0  # Se todos são empates
                    except:
                        p_val = 1.0
                    
                    raw_pvalues[alg1][alg2] = p_val
                    
                    if i < j:  # Evitar duplicatas
                        comparisons.append({
                            'algorithm1': alg1,
                            'algorithm2': alg2,
                            'raw_pvalue': p_val
                        })
                else:
                    raw_pvalues[alg1][alg2] = np.nan
        
        return comparisons, raw_pvalues
    
    def _holm_correction(self, pvalues):
        """
        Correção de Holm
        """
        n = len(pvalues)
        # Ordenar p-values com índices
        sorted_pairs = sorted(enumerate(pvalues), key=lambda x: x[1])
        
        corrected = [0] * n
        for i, (original_idx, p_val) in enumerate(sorted_pairs):
            correction_factor = n - i
            corrected_p = min(p_val * correction_factor, 1.0)
            
            # Garantir monotonicidade
            if i > 0:
                corrected_p = max(corrected_p, corrected[sorted_pairs[i-1][0]])
            
            corrected[original_idx] = corrected_p
        
        return corrected
    
    def post_hoc_test(self, correction='holm'):
        """
        Executa testes post-hoc com correção para múltiplas comparações
        """
        comparisons, raw_pvalues_dict = self._pairwise_comparisons()
        
        # Extrair apenas os p-values para correção
        raw_pvals = [comp['raw_pvalue'] for comp in comparisons]
        
        # Aplicar correção de Holm
        corrected_pvals = self._holm_correction(raw_pvals)
        
        # Adicionar p-values corrigidos às comparações
        for i, comp in enumerate(comparisons):
            comp['corrected_pvalue'] = corrected_pvals[i]
        
        # Criar matriz de p-values corrigidos
        corrected_pvalues_matrix = {}
        for alg1 in self.algorithm_names:
            corrected_pvalues_matrix[alg1] = {}
            for alg2 in self.algorithm_names:
                if alg1 == alg2:
                    corrected_pvalues_matrix[alg1][alg2] = np.nan
                else:
                    # Encontrar a comparação correspondente
                    for comp in comparisons:
                        if ((comp['algorithm1'] == alg1 and comp['algorithm2'] == alg2) or
                            (comp['algorithm1'] == alg2 and comp['algorithm2'] == alg1)):
                            corrected_pvalues_matrix[alg1][alg2] = comp['corrected_pvalue']
                            break
        
        return {
            'summary': dict(self.mean_ranks),
            'raw_pval': raw_pvalues_dict,
            'corrected_pval': corrected_pvalues_matrix,
            'comparisons': comparisons,
            'correction_method': correction
        }
    
    def full_analysis(self, alpha=0.05):
        """
        Executa análise completa: teste omnibus + post-hoc se necessário
        """
        results = {}
        
        # Teste omnibus (Iman-Davenport)
        omnibus_result = self.iman_davenport_test()
        results['omnibus_test'] = omnibus_result
        
        print(f"\n{omnibus_result['test_name']}")
        print(f"F-statistic = {omnibus_result['statistic']:.4f}")
        print(f"Chi-squared = {omnibus_result['chi2_statistic']:.4f}")
        print(f"df1 = {omnibus_result['df1']}, df2 = {omnibus_result['df2']}")
        print(f"p-value = {omnibus_result['p_value']:.2e}")
        
        # Se significativo, executar post-hoc
        if omnibus_result['p_value'] < alpha:
            print(f"\nP-value < {alpha}, executando testes post-hoc...")
            
            post_hoc_result = self.post_hoc_test()
            results['post_hoc_test'] = post_hoc_result
            
            print(f"\nRanks médios:")
            for alg, rank in post_hoc_result['summary'].items():
                print(f"{alg}: {rank:.4f}")
            
            print(f"\nP-values corrigidos (correção de Holm):")
            corrected_df = pd.DataFrame(post_hoc_result['corrected_pval'])
            print(corrected_df.round(6))
            
        else:
            print(f"\nP-value >= {alpha}, não há diferenças significativas entre os algoritmos.")
            results['post_hoc_test'] = None
        
        return results

def main():
    print("=== TESTE DE FRIEDMAN - IMPLEMENTAÇÃO FINAL ===")
    print("Usando scipy + metodologia scmamp do R")
    print("=" * 50)
    
    # Carregar dados
    print("\n1. Carregando dados...")
    df = pd.read_excel('./results.xlsx')
    data = df.drop(columns=['Unnamed: 0'])
    
    print(f"Dados: {data.shape[0]} datasets, {data.shape[1]} algoritmos")
    print(f"Algoritmos: {list(data.columns)}")
    
    # Executar análise
    print("\n2. Executando análise...")
    friedman = FriedmanTestFinal(data)
    
    print(f"\nRankings médios:")
    for alg, rank in friedman.mean_ranks.items():
        print(f"{alg:15}: {rank:.4f}")
    
    results = friedman.full_analysis()
    
    # Salvar resultados
    print("\n3. Salvando resultados...")
    
    # Relatório
    report = []
    report.append("TESTE DE FRIEDMAN - RESULTADOS FINAIS")
    report.append("=" * 50)
    report.append("")
    report.append(f"Datasets analisados: {data.shape[0]}")
    report.append(f"Algoritmos: {', '.join(data.columns)}")
    report.append("")
    
    # Rankings
    report.append("RANKINGS MÉDIOS (1 = melhor, 4 = pior):")
    sorted_algs = sorted(friedman.mean_ranks.items(), key=lambda x: x[1])
    for i, (alg, rank) in enumerate(sorted_algs, 1):
        report.append(f"{i}º: {alg} (rank: {rank:.4f})")
    report.append("")
    
    # Teste omnibus
    omnibus = results['omnibus_test']
    report.append("TESTE OMNIBUS:")
    report.append(f"Chi-squared: {omnibus['chi2_statistic']:.4f}")
    report.append(f"F-statistic: {omnibus['statistic']:.4f}")
    report.append(f"p-value: {omnibus['p_value']:.2e}")
    
    if omnibus['p_value'] < 0.05:
        report.append("CONCLUSÃO: Diferenças significativas detectadas")
    else:
        report.append("CONCLUSÃO: Não há diferenças significativas")
    
    # Post-hoc se aplicável
    if results['post_hoc_test']:
        report.append("")
        report.append("TESTES POST-HOC:")
        corrected_df = pd.DataFrame(results['post_hoc_test']['corrected_pval'])
        report.append(corrected_df.round(6).to_string())
    
    # Salvar
    with open('./friedman_final_results.txt', 'w') as f:
        f.write('\n'.join(report))
    
    # Excel
    with pd.ExcelWriter('./friedman_final_results.xlsx') as writer:
        data.to_excel(writer, sheet_name='Dados', index=True)
        friedman.ranks.to_excel(writer, sheet_name='Rankings', index=True)
        
        # Rankings médios
        ranks_df = pd.DataFrame({
            'Algoritmo': [alg for alg, _ in sorted_algs],
            'Rank_Medio': [rank for _, rank in sorted_algs],
            'Posicao': range(1, len(sorted_algs) + 1)
        })
        ranks_df.to_excel(writer, sheet_name='Rankings_Medios', index=False)
        
        # Teste omnibus
        omnibus_df = pd.DataFrame([{
            'Teste': 'Iman-Davenport',
            'Chi2': omnibus['chi2_statistic'],
            'F_stat': omnibus['statistic'],
            'p_value': omnibus['p_value'],
            'Significativo': omnibus['p_value'] < 0.05
        }])
        omnibus_df.to_excel(writer, sheet_name='Teste_Omnibus', index=False)
        
        if results['post_hoc_test']:
            corrected_df.to_excel(writer, sheet_name='Post_Hoc', index=True)
    
    print("Arquivos salvos:")
    print("- friedman_final_results.txt")
    print("- friedman_final_results.xlsx")
    
    print("\n" + "=" * 50)
    print("RESUMO FINAL:")
    print("=" * 50)
    for i, (alg, rank) in enumerate(sorted_algs, 1):
        print(f"{i}º lugar: {alg} (rank médio: {rank:.3f})")
    
    if omnibus['p_value'] < 0.05:
        print(f"\n✓ Há diferenças significativas (p = {omnibus['p_value']:.2e})")
    else:
        print(f"\n✗ Não há diferenças significativas (p = {omnibus['p_value']:.2e})")
    
    return results

if __name__ == "__main__":
    results = main()


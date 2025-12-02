import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="Simulador ENEM",
    page_icon="🎓",
    layout="centered"
)

# Título
st.title("🎓 Simulador de Notas ENEM")
st.write("Descubra suas notas previstas baseado no seu perfil!")

# Carregar modelo
@st.cache_resource
def load_model():
    try:
        modelo_completo = joblib.load('modelo_enem_final.pkl')
        return modelo_completo['modelos'], modelo_completo['scaler'], modelo_completo['colunas_treino']
    except:
        st.error("❌ Erro: Arquivo do modelo não encontrado!")
        return None, None, None

modelos, scaler, colunas_treino = load_model()

if modelos is not None:
    # Formulário
    st.header("📋 Preencha seus dados:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        idade = st.selectbox("🎂 Idade:", [
            (1, "Menor de 17 anos"),
            (2, "17 anos"),
            (3, "18 anos"),
            (4, "19 anos"),
            (5, "20 anos"),
            (6, "21 anos"),
            (7, "22 anos"),
            (8, "23 anos"),
            (11, "26-30 anos"),
            (12, "31-35 anos")
        ], format_func=lambda x: x[1], index=2)
        
        escola = st.selectbox("🏫 Tipo de Escola:", [
            (1, "Privada"),
            (2, "Pública")
        ], format_func=lambda x: x[1], index=1)

        
        raca = st.selectbox("🧑 Cor/Raça:", [
            (0, "Não declarado"),
            (1, "Branca"),
            (2, "Preta"),
            (3, "Parda"),
            (4, "Amarela"),
            (5, "Indígena")
        ], format_func=lambda x: x[1], index=3)
    
    with col2:
        estado = st.selectbox("🌎 Estado:", [
            "SP", "RJ", "MG", "BA", "PR", "RS", "PE", "CE", "SC", "GO"
        ])
        
        renda = st.selectbox("💰 Renda Familiar:", [
            ("A", "Nenhuma renda"),
            ("B", "Até R$ 1.320"),
            ("C", "R$ 1.320 - R$ 1.980"),
            ("D", "R$ 1.980 - R$ 2.640"),
            ("E", "R$ 2.640 - R$ 3.300"),
            ("F", "R$ 3.300 - R$ 3.960"),
            ("G", "R$ 3.960 - R$ 5.280"),
            ("H", "R$ 5.280 - R$ 6.600"),
            ("Q", "Acima de R$ 26.400")
        ], format_func=lambda x: x[1], index=2)
    
    # Botão de predição
    if st.button("🔮 Simular Notas", type="primary"):
        # Processar dados
        dados_aluno = {
            'TP_FAIXA_ETARIA': idade[0],
            'TP_ESCOLA': escola[0],
            'TP_COR_RACA': raca[0],
            'SG_UF_PROVA': estado,
            'Q006': renda[0]
        }
        
        df_aluno = pd.DataFrame([dados_aluno])
        
        # Converter para categórico
        cat_cols = ['TP_FAIXA_ETARIA','TP_COR_RACA','TP_ESCOLA','SG_UF_PROVA','Q006']
        for col in cat_cols:
            df_aluno[col] = df_aluno[col].astype('category')
        
        # One-hot encoding
        df_encoded = pd.get_dummies(df_aluno)
        df_final = df_encoded.reindex(columns=colunas_treino, fill_value=0)
        
        # Escalar dados
        df_scaled = scaler.transform(df_final)
        
        # Fazer predições
        materias = ['Ciências Natureza', 'Ciências Humanas', 'Linguagens', 'Matemática', 'Redação']
        
        st.header("📊 Suas Notas Previstas:")
        
        soma = 0
        for i, materia in enumerate(materias):
            modelo = modelos[materia]
            previsao = modelo.predict(df_scaled)
            nota = previsao[0][i] if previsao.ndim > 1 else previsao[i]
            nota = round(nota, 1)
            soma += nota
            
            # Mostrar nota com cor baseada no valor
            if nota >= 700:
                st.success(f"📚 **{materia}**: {nota}")
            elif nota >= 500:
                st.info(f"📚 **{materia}**: {nota}")
            else:
                st.warning(f"📚 **{materia}**: {nota}")
        
        media = round(soma/5, 1)
        st.metric("📈 Média Geral", f"{media}", delta=None)
        
        # Interpretação
        if media >= 700:
            st.balloons()
            st.success("🎉 Excelente! Notas muito boas para universidades concorridas!")
        elif media >= 500:
            st.info("👍 Bom desempenho! Você tem chances em várias universidades!")
        else:
            st.warning("📖 Continue estudando! Há potencial para melhorar!")

else:
    st.error("Não foi possível carregar o modelo. Verifique se o arquivo 'modelo_enem_final.pkl' está presente.")

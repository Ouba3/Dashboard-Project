# ============================================================================
# DASHBOARD ESPÉRANCE DE VIE - STREAMLIT
# À exécuter dans VSCode avec : streamlit run app.py
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Dashboard Espérance de Vie",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================
@st.cache_data
def load_data():
    """
    Charge les données depuis le fichier Excel
    """
    try:
        # Charger depuis Excel
        df = pd.read_excel('Donnees_finales.xlsx')
        
        # Renommer les colonnes si nécessaire
        column_mapping = {
            "Period life expectancy at birth": "Life_expectancy",
            "Life expectancy - Sex: female - Age: 0 - Variant: estimates": "LE_Female",
            "Life expectancy - Sex: male - Age: 0 - Variant: estimates": "LE_Male",
            "Population - Sex: all - Age: all - Variant: estimates": "Population",
            "Entity": "country",
            "Year": "year",
            "Code": "code"
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Créer la colonne 'code' si elle n'existe pas (codes ISO)
        if 'code' not in df.columns and 'Code' in df.columns:
            df['code'] = df['Code']
        
        # Calculer l'écart F/H si pas déjà fait
        if 'LE_gap_FH' not in df.columns and 'LE_Female' in df.columns and 'LE_Male' in df.columns:
            df["LE_gap_FH"] = df["LE_Female"] - df["LE_Male"]
        
        # Formater la population
        if 'Population' in df.columns:
            df["Population_formatted"] = df["Population"].apply(
                lambda x: f"{int(x):,}".replace(",", " ") if pd.notna(x) else "N/A"
            )
        
        return df
    
    except FileNotFoundError:
        st.error("❌ Fichier 'Donnees_finales.xlsx' introuvable. Assurez-vous qu'il est dans le même dossier que app.py")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement : {e}")
        st.stop()

# Chargement
with st.spinner("🔄 Chargement des données..."):
    df = load_data()

st.success(f"✅ {len(df):,} observations | {df['year'].nunique()} années | {df['country'].nunique()} pays")

# ============================================================================
# SIDEBAR - FILTRES GLOBAUX
# ============================================================================
st.sidebar.title("🎛️ Filtres")

# Filtre période
years = sorted(df['year'].unique())
year_range = st.sidebar.slider(
    "📅 Période",
    min_value=int(min(years)),
    max_value=int(max(years)),
    value=(int(min(years)), int(max(years))),
    step=1,
    key="sidebar_year_range"
)

# Filtre continent
continents = ['Tous'] + sorted(df['Continent'].dropna().unique().tolist())
selected_continent = st.sidebar.selectbox("🌍 Continent", continents)

# Filtre pays (dépend du continent)
if selected_continent == 'Tous':
    countries = sorted(df['country'].dropna().unique().tolist())
else:
    countries = sorted(df[df['Continent'] == selected_continent]['country'].dropna().unique().tolist())

selected_countries = st.sidebar.multiselect(
    "🏳️ Pays (optionnel, max 5)",
    countries,
    max_selections=5
)

# Appliquer les filtres
filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
if selected_continent != 'Tous':
    filtered_df = filtered_df[filtered_df['Continent'] == selected_continent]
if selected_countries:
    filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]

st.sidebar.markdown("---")
st.sidebar.info(f"📊 **{len(filtered_df):,}** observations")

# ============================================================================
# EN-TÊTE
# ============================================================================
st.title("🌍 Dashboard Espérance de Vie Mondiale")
st.markdown("""
**Problématique** : *Comment l'espérance de vie à la naissance a-t-elle évolué selon les continents, 
les pays et le sexe au fil du temps, et quel est son lien avec le PIB ?*
""")
st.markdown("---")

# ============================================================================
# NAVIGATION PAR ONGLETS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Vue d'ensemble",
    "⚧ Analyse Genre",
    "💰 Relation PIB",
    "📈 Modèle de Régression"
])

# ============================================================================
# TAB 1 : VUE D'ENSEMBLE
# ============================================================================
with tab1:
    st.header("📊 Vue d'ensemble")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_le = filtered_df['Life_expectancy'].mean()
        st.metric("📊 EV moyenne", f"{avg_le:.1f} ans")
    
    with col2:
        avg_male = filtered_df['LE_Male'].mean()
        st.metric("👨 Hommes", f"{avg_male:.1f} ans")
    
    with col3:
        avg_female = filtered_df['LE_Female'].mean()
        st.metric("👩 Femmes", f"{avg_female:.1f} ans")
    
    with col4:
        gap = filtered_df['LE_gap_FH'].mean()
        st.metric("⚖️ Écart F/H", f"{gap:.1f} ans")
    
    st.markdown("---")
    
    # ========================================================================
    # KPI 0 : CARTE INTERACTIVE MONDIALE
    # ========================================================================
    st.subheader("🌍 Carte interactive - Espérance de vie mondiale")
    
    # Préparation des données
    df_map = filtered_df.copy()
    df_map = df_map.dropna(subset=["country", "Life_expectancy", "year"])
    
    # Options utilisateur
    col_a, col_b = st.columns([3, 1])
    with col_b:
        show_animation_map = st.checkbox(
            "Animation temporelle",
            value=True,
            key="anim_map"
        )
        
        if not show_animation_map:
            selected_year_map = st.slider(
                "Année affichée",
                min_value=int(df_map["year"].min()),
                max_value=int(df_map["year"].max()),
                value=int(df_map["year"].max()),
                key="year_map"
            )
    
    # Graphique
    if show_animation_map:
        # Version animée avec locationmode="country names"
        fig_map = px.choropleth(
            df_map,
            locations="country",
            locationmode="country names",
            color="Life_expectancy",
            hover_name="country",
            animation_frame="year",
            color_continuous_scale="Viridis",
            projection="natural earth",
            labels={"Life_expectancy": "Espérance de vie (ans)"},
            title=""
        )
    else:
        # Version statique pour l'année sélectionnée
        df_year_map = df_map[df_map["year"] == selected_year_map]
        fig_map = px.choropleth(
            df_year_map,
            locations="country",
            locationmode="country names",
            color="Life_expectancy",
            hover_name="country",
            color_continuous_scale="Viridis",
            projection="natural earth",
            labels={"Life_expectancy": "Espérance de vie (ans)"},
            title=f"Espérance de vie par pays en {selected_year_map}"
        )
    
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        coloraxis_colorbar_title="Années"
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # NOUVEAUX INSIGHTS POUR LA CARTE
    with st.expander("💡 Insights clés - Répartition géographique"):
        col1, col2, col3 = st.columns(3)
        
        # Pays avec la plus haute EV
        max_le_country = df_map.loc[df_map['Life_expectancy'].idxmax()]
        with col1:
            st.metric(
                "🏆 Espérance de vie maximale",
                f"{max_le_country['Life_expectancy']:.1f} ans",
                f"{max_le_country['country']}"
            )
        
        # Pays avec la plus basse EV
        min_le_country = df_map.loc[df_map['Life_expectancy'].idxmin()]
        with col2:
            st.metric(
                "⚠️ Espérance de vie minimale",
                f"{min_le_country['Life_expectancy']:.1f} ans",
                f"{min_le_country['country']}"
            )
        
        # Écart géographique
        with col3:
            geo_gap = max_le_country['Life_expectancy'] - min_le_country['Life_expectancy']
            st.metric(
                "📏 Écart géographique",
                f"{geo_gap:.1f} ans",
                "Entre max et min"
            )
    
    st.markdown("---")

      # ========================================================================
    # KPI 1 : Heatmap dynamique de l'espérance de vie par continent
    # ========================================================================
    # ========================================================================
# TAB : HEATMAP PAR CONTINENT
# ========================================================================
with tab1:  # tu peux mettre dans tab1 ou créer un tab spécifique
    st.subheader("🌍 Heatmap dynamique de l'espérance de vie par continent")

    # --- Nettoyage et renommage des colonnes ---
    df_heatmap = df.copy()
    df_heatmap.columns = df_heatmap.columns.str.strip()  # enlever espaces invisibles
    column_mapping = {
        "Period life expectancy at birth": "Life_expectancy",
        "Life expectancy - Sex: female - Age: 0 - Variant: estimates": "LE_Female",
        "Life expectancy - Sex: male - Age: 0 - Variant: estimates": "LE_Male",
        "Population - Sex: all - Age: all - Variant: estimates": "Population",
        "Entity": "country",
        "Year": "year",
        "Code": "code"
    }
    df_heatmap = df_heatmap.rename(columns=column_mapping)

    # --- Grouper par continent et année ---
    df_grouped = (
        df_heatmap.groupby(['Continent', 'year'])
                  .agg({'Life_expectancy': 'mean'})
                  .reset_index()
    )

    # --- Pivot pour heatmap ---
    df_pivot = df_grouped.pivot(
        index='Continent',
        columns='year',
        values='Life_expectancy'
    )

    # --- Heatmap Plotly Express simple et rapide ---
    fig_heatmap = px.imshow(
        df_pivot,
        labels=dict(x="Année", y="Continent", color="Espérance de vie (ans)"),
        text_auto=True,
        aspect="auto",
        color_continuous_scale='YlGnBu'
    )

    # --- Afficher le graphique dans Streamlit ---
    st.plotly_chart(fig_heatmap, width='stretch')   

    # NOUVEAUX INSIGHTS POUR LA HEATMAP
    with st.expander("💡 Insights clés - Évolution par continent"):
        col1, col2, col3 = st.columns(3)
        
        # Continent avec la plus forte progression
        first_year_data = df_grouped[df_grouped['year'] == df_grouped['year'].min()].set_index('Continent')['Life_expectancy']
        last_year_data = df_grouped[df_grouped['year'] == df_grouped['year'].max()].set_index('Continent')['Life_expectancy']
        progression = (last_year_data - first_year_data).sort_values(ascending=False)
        
        with col1:
            st.metric(
                "🚀 Plus forte progression",
                f"+{progression.iloc[0]:.1f} ans",
                progression.index[0]
            )
        
        with col2:
            st.metric(
                "🐌 Plus faible progression",
                f"+{progression.iloc[-1]:.1f} ans",
                progression.index[-1]
            )
        
        with col3:
            avg_progression = progression.mean()
            st.metric(
                "📊 Progression moyenne",
                f"+{avg_progression:.1f} ans",
                "Tous continents"
            ) 
    # ========================================================================
    # KPI 1 : ESPÉRANCE DE VIE MONDIALE PONDÉRÉE
    # ========================================================================
    st.subheader("📈  Espérance de vie moyenne mondiale (pondérée par la population)")
    
    # Calcul de la moyenne pondérée
    world_weighted = (
        filtered_df
        .dropna(subset=["Life_expectancy", "Population"])
        .groupby("year")
        .apply(lambda g: np.average(g["Life_expectancy"], weights=g["Population"]), include_groups=False)
        .reset_index(name="life_exp_world_weighted")
    )
    
    # Graphique
    fig_world = px.line(
        world_weighted,
        x="year",
        y="life_exp_world_weighted",
        title="Espérance de vie moyenne mondiale (pondérée par la population)",
        labels={"year": "Année", "life_exp_world_weighted": "Espérance de vie (ans)"},
        markers=True
    )
    
    fig_world.update_layout(
        height=500,
        hovermode='x unified'
    )
    
    fig_world.update_traces(
        line=dict(color='#2ecc71', width=3),
        marker=dict(size=6)
    )
    
    st.plotly_chart(fig_world, use_container_width=True)
    
    # Insights
    with st.expander("💡 Insights clés"):
        if len(world_weighted) > 0:
            col1, col2, col3 = st.columns(3)
            
            first_year = world_weighted.iloc[0]
            last_year = world_weighted.iloc[-1]
            gain_total = last_year["life_exp_world_weighted"] - first_year["life_exp_world_weighted"]
            
            with col1:
                st.metric(
                    f"📅 {int(first_year['year'])}",
                    f"{first_year['life_exp_world_weighted']:.1f} ans"
                )
            
            with col2:
                st.metric(
                    f"📅 {int(last_year['year'])}",
                    f"{last_year['life_exp_world_weighted']:.1f} ans"
                )
            
            with col3:
                st.metric(
                    "📈 Gain total",
                    f"+{gain_total:.1f} ans"
                )
    
    st.markdown("---")
    
    # ========================================================================
    # KPI 2 : TOP 10 PAYS - PLUS HAUTE ET PLUS BASSE ESPÉRANCE DE VIE
    # ========================================================================
    st.subheader("🏆 Top 10 des pays par espérance de vie")

    # Nom exact de la colonne PIB
    GDP_COL = "GDP per capita"

    # On garde seulement les lignes valides
    base_df = filtered_df.dropna(subset=["Life_expectancy", "country"]).copy()

    # ---------- TOP 10 PLUS HAUTE ESPÉRANCE DE VIE (toutes années) ----------
    top10_highest_all = (
        base_df
        .sort_values(["year", "Life_expectancy"], ascending=[True, False])
        .groupby("year")
        .head(10)
    )

    # ---------- TOP 10 PLUS FAIBLE ESPÉRANCE DE VIE (toutes années) ----------
    top10_lowest_all = (
        base_df
        .sort_values(["year", "Life_expectancy"], ascending=[True, True])
        .groupby("year")
        .head(10)
    )

    # Limites X cohérentes pour ne pas bouger pendant l'animation
    min_x = base_df["Life_expectancy"].min() - 1
    max_x = base_df["Life_expectancy"].max() + 1

    # Affichage en deux colonnes
    col_high, col_low = st.columns(2)

    # ---------- GRAPH HIGH ----------
    with col_high:
        st.markdown("**🟢 Top 10 - Espérance de vie la plus élevée (animation)**")
        st.caption("Survolez les barres pour voir le PIB par habitant 💶")

        fig_high = px.bar(
            top10_highest_all,
            x="Life_expectancy",
            y="country",
            orientation="h",
            color="Life_expectancy",
            color_continuous_scale="Greens",
            labels={
                "Life_expectancy": "Années",
                "country": "Pays",
                GDP_COL: "PIB par habitant"
            },
            text="Life_expectancy",
            animation_frame="year",           # ▶️ bouton Play
            range_x=[min_x, max_x],

            hover_data={
                "Life_expectancy":":.1f",
                GDP_COL:":,.0f",               # 💰 PIB au survol
                "year": True,
                "country": False
            },
        )

        fig_high.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_high.update_layout(
            showlegend=False,
            height=420,
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=0, r=0, t=0, b=0),
        )
        fig_high.update_coloraxes(showscale=False)

        st.plotly_chart(fig_high, use_container_width=True)

    # ---------- GRAPH LOW ----------
    with col_low:
        st.markdown("**🔴 Top 10 - Espérance de vie la plus faible (animation)**")
        st.caption("Survolez les barres pour voir le PIB par habitant 💶")

        fig_low = px.bar(
            top10_lowest_all,
            x="Life_expectancy",
            y="country",
            orientation="h",
            color="Life_expectancy",
            color_continuous_scale="Reds",
            labels={
                "Life_expectancy": "Années",
                "country": "Pays",
                GDP_COL: "PIB par habitant"
            },
            text="Life_expectancy",
            animation_frame="year",
            range_x=[min_x, max_x],

            hover_data={
                "Life_expectancy":":.1f",
                GDP_COL:":,.0f",
                "year": True,
                "country": False
            },
        )

        fig_low.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_low.update_layout(
            showlegend=False,
            height=420,
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=0, r=0, t=0, b=0),
        )
        fig_low.update_coloraxes(showscale=False)

        st.plotly_chart(fig_low, use_container_width=True)

    # ---------- STATISTIQUES COMPARATIVES ----------
    with st.expander("📊 Statistiques comparatives"):

        available_years = sorted(base_df["year"].unique())
        last_year = available_years[-1]

        df_last_year = base_df[base_df["year"] == last_year].copy()

        top10_highest_last = df_last_year.nlargest(
            10, "Life_expectancy"
        )[["country", "Life_expectancy", GDP_COL]]

        top10_lowest_last = df_last_year.nsmallest(
            10, "Life_expectancy"
        )[["country", "Life_expectancy", GDP_COL]]

        col1, col2, col3 = st.columns(3)

        with col1:
            highest_country = top10_highest_last.iloc[0]
            st.metric(
                f"🥇 Pays #1 ({last_year})",
                highest_country["country"],
                f"{highest_country['Life_expectancy']:.1f} ans"
            )
            st.caption(f"PIB/hab : {highest_country[GDP_COL]:,.0f}")

        with col2:
            lowest_country = top10_lowest_last.iloc[0]
            st.metric(
                f"❌ Pays dernier ({last_year})",
                lowest_country["country"],
                f"{lowest_country['Life_expectancy']:.1f} ans"
            )
            st.caption(f"PIB/hab : {lowest_country[GDP_COL]:,.0f}")

        with col3:
            gap = highest_country["Life_expectancy"] - lowest_country["Life_expectancy"]
            st.metric(
                "📏 Écart max d'espérance de vie",
                f"{gap:.1f} ans",
                "Entre #1 et dernier"
            )
            st.caption("Le PIB est consultable via le survol des barres.")
# ============================================================================
# TAB 2 : ANALYSE GENRE
# ============================================================================
with tab2:
    st.header("⚧ Analyse de l'écart Femmes/Hommes")
    
    # Agrégation par continent / année
    df_continent_gender = filtered_df.groupby(["Continent", "year"]).agg(
        LE_gap_FH=("LE_gap_FH", "mean"),
        Population=("Population", "sum")
    ).reset_index()
    
    # Colonnes formatées pour le hover
    df_continent_gender["GAP"] = df_continent_gender["LE_gap_FH"].round(2)
    df_continent_gender["Population_hover"] = df_continent_gender["Population"].apply(
        lambda x: f"{int(x):,}".replace(",", " ") if pd.notna(x) else "N/A"
    )
    
    # Bar chart horizontal animé
    fig2 = px.bar(
        df_continent_gender,
        x="LE_gap_FH",
        y="Continent",
        color="LE_gap_FH",
        orientation="h",
        animation_frame="year",
        hover_name="Continent",
        hover_data={
            "GAP": True,
            "Population_hover": True,
            "LE_gap_FH": False,
            "Population": False
        },
        color_continuous_scale="RdBu_r",
        title="Écart de longévité Femmes-Hommes par continent"
    )
    
    fig2.update_layout(
        xaxis_title="Écart Femmes - Hommes (années)",
        yaxis_title="Continent",
        coloraxis_colorbar_title="Écart F/H",
        height=500
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Insights
    with st.expander("💡 Insights clés"):
        avg_gap = filtered_df['LE_gap_FH'].mean()
        max_gap = filtered_df.groupby('Continent')['LE_gap_FH'].mean().max()
        max_continent = filtered_df.groupby('Continent')['LE_gap_FH'].mean().idxmax()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Écart moyen global", f"{avg_gap:.2f} ans")
        with col2:
            st.metric(f"Plus grand écart ({max_continent})", f"{max_gap:.2f} ans")

# ============================================================================
# TAB 3 : RELATION PIB
# ============================================================================
with tab3:
    st.header("💰 Relation entre PIB et Espérance de Vie")
    
   # ========================================================================
# KPI : Évolution du PIB moyen par continent
# ========================================================================

    # --- Titre ou sous-section du dashboard ---
    st.subheader("💰 Évolution du PIB moyen par continent")

    # --- Préparer les données ---
    df_plot = df.copy()  # utiliser directement df_o avec les colonnes déjà renommées

    # --- Calcul du PIB moyen par continent ---
    records = []
    for continent in df_plot['Continent'].dropna().unique():
        df_cont = df_plot[df_plot['Continent'] == continent]
        for year, group in df_cont.groupby('year'):  # utiliser 'year' directement
            GDP_mean = group[GDP_COL].mean()
            records.append({'Continent': continent, 'year': year, 'GDP_mean': GDP_mean})

    df_mean = pd.DataFrame(records)

    # --- Graphique interactif Plotly ---
    fig = px.line(
        df_mean,
        x='year',  # utiliser 'year' directement
        y='GDP_mean',
        color='Continent',
        markers=True,
        line_shape='spline',  # lignes lisses
        color_discrete_sequence=px.colors.qualitative.Bold,  # palette colorée
        labels={'GDP_mean': 'PIB moyen (USD)', 'year': 'Année', 'Continent': 'Continent'},
        title=""
    )

    # Hover plus clair
    fig.update_traces(hovertemplate='Année: %{x}<br>PIB moyen: %{y:$,.0f}<br>Continent: %{color}')
    fig.update_layout(hovermode='x unified', height=600)

    # --- Affichage dans Streamlit ---
    st.plotly_chart(fig, width='stretch')

    # INSIGHTS POUR LE PIB PAR CONTINENT
    with st.expander("💡 Insights clés - Disparités économiques par continent"):
        col1, col2, col3 = st.columns(3)
        
        # Continent le plus riche et plus pauvre
        latest_gdp = df_mean[df_mean['year'] == df_mean['year'].max()]
        richest = latest_gdp.loc[latest_gdp['GDP_mean'].idxmax()]
        poorest = latest_gdp.loc[latest_gdp['GDP_mean'].idxmin()]
        
        with col1:
            st.metric(
                "💎 Continent le plus riche",
                richest['Continent'],
                f"${richest['GDP_mean']:,.0f}"
            )
        
        with col2:
            st.metric(
                "📉 Continent le plus pauvre",
                poorest['Continent'],
                f"${poorest['GDP_mean']:,.0f}"
            )
        
        with col3:
            ratio = richest['GDP_mean'] / poorest['GDP_mean']
            st.metric(
                "⚖️ Ratio de richesse",
                f"×{ratio:.1f}",
                "Plus riche / plus pauvre"
            )
        
        st.markdown("---")
        st.markdown("**📊 Croissance économique par continent**")
        
        # Calcul des taux de croissance
        growth_data = []
        for continent in df_mean['Continent'].unique():
            cont_data = df_mean[df_mean['Continent'] == continent].sort_values('year')
            if len(cont_data) > 1:
                first_gdp = cont_data.iloc[0]['GDP_mean']
                last_gdp = cont_data.iloc[-1]['GDP_mean']
                growth_pct = ((last_gdp - first_gdp) / first_gdp) * 100
                growth_data.append({'Continent': continent, 'Croissance': growth_pct})
        
        growth_df = pd.DataFrame(growth_data).sort_values('Croissance', ascending=False)
        st.dataframe(growth_df.style.format({'Croissance': '{:.1f}%'}), hide_index=True)
    
    st.markdown("---")

    # ========================================================================
    # KPI : GAIN MARGINAL (BETA)
    # ========================================================================
    st.subheader("📈 Sensibilité de la longévité au PIB par pays et par année")
    
    # Préparer les données
    df_plot_beta = filtered_df.copy()
    df_plot_beta["Population"] = pd.to_numeric(
        df_plot_beta["Population"].astype(str).str.replace(",", ""), errors="coerce"
    )
    
    window = 5
    years_list = sorted(df_plot_beta['year'].unique())
    betas = []
    
    # Calcul beta glissant par pays
    for entity, group in df_plot_beta.groupby("country"):
        group = group.dropna(subset=["Life_expectancy", "GDP per capita", "Population"])
        group = group[group["GDP per capita"] > 0]
        
        for year in years_list:
            group_window = group[(group['year'] >= year - window//2) & (group['year'] <= year + window//2)]
            if len(group_window) < 3:
                continue
            
            X = np.log(group_window["GDP per capita"])
            X = sm.add_constant(X)
            y = group_window["Life_expectancy"]
            
            try:
                model = sm.OLS(y, X).fit()
                beta = model.params[1]
                betas.append({
                    "country": entity,
                    "year": year,
                    "Beta": beta,
                    "Population": group_window["Population"].mean(),
                    "GDP per capita": group_window["GDP per capita"].mean()
                })
            except:
                continue
    
    if betas:
        df_beta = pd.DataFrame(betas)
        df_beta["Population size"] = df_beta["Population"].apply(lambda x: f"{int(x):,}".replace(",", " "))
        
        # Options utilisateur
        col_a, col_b = st.columns([3,1])
        with col_b:
            show_animation_beta = st.checkbox("Animation temporelle", value=True, key="anim_beta")
            selected_year_beta = st.slider(
                "Année affichée",
                int(df_beta['year'].min()),
                int(df_beta['year'].max()),
                int(df_beta['year'].max()),
                key="year_beta"
            )
        
        # Filtrer pour graphique
        df_plot_chart = df_beta if show_animation_beta else df_beta[df_beta["year"] == selected_year_beta]
        
        # Graphique animé ou statique
        fig3 = px.scatter(
            df_plot_chart,
            x="GDP per capita",
            y="Beta",
            size="Population",
            color="Beta",
            hover_name="country",
            hover_data={
                "Beta": ":.3f",
                "Population size": True,
                "GDP per capita": ":,.0f"
            },
            log_x=True,
            size_max=60,
            color_continuous_scale=["#ffe6f0", "#ff66b3", "#ff0066"],
            animation_frame="year" if show_animation_beta else None
        )
        
        fig3.update_layout(
            xaxis_title="PIB par habitant (log)",
            yaxis_title="Beta : sensibilité EV au PIB",
            height=600
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Explications
        with st.expander("💡 Interprétation du Beta"):
            st.markdown("""
            Le **coefficient Beta** mesure la sensibilité de l'espérance de vie au PIB :
            - **Beta positif** : Une augmentation du PIB est associée à une hausse de l'espérance de vie
            - **Beta élevé** : Le pays améliore fortement sa longévité avec la richesse
            - **Beta faible** : L'impact du PIB sur la longévité est limité
            """)
    else:
        st.warning("⚠️ Pas assez de données pour calculer les betas avec les filtres actuels")
    
    st.markdown("---")
    
    # ========================================================================
    # KPI : ÉVOLUTION EV EN FONCTION DU PIB
    # ========================================================================
    st.subheader("📊 Évolution de l'espérance de vie en fonction du PIB par habitant")
    
    # Préparation des données
    df_plot = filtered_df.copy()
    df_plot["Population"] = df_plot["Population"].apply(
        lambda x: int(str(x).replace(",", "")) if pd.notna(x) else 0
    )
    
    df_plot = df_plot.dropna(subset=["Life_expectancy", "GDP per capita"])
    df_plot = df_plot[df_plot["GDP per capita"] > 0]
    
    # Segmentation du PIB par quantiles
    df_plot["GDP_group"] = df_plot.groupby("year")["GDP per capita"].transform(
        lambda x: pd.qcut(x, q=3, labels=["Faible revenu", "Intermédiaire", "Haut revenu"], duplicates='drop')
    )
    
    df_plot["Population size"] = df_plot["Population"].apply(lambda x: f"{x:,}".replace(",", " "))
    
    # Options d'affichage
    col_a, col_b = st.columns([3, 1])
    with col_b:
        show_animation = st.checkbox("Animation temporelle", value=True, key="anim1")
        selected_year_display = st.slider(
            "Année affichée",
            min_value=int(df_plot['year'].min()),
            max_value=int(df_plot['year'].max()),
            value=int(df_plot['year'].max()),
            key="year1"
        )
    
    # Graphique
    if show_animation:
        fig1 = px.scatter(
            df_plot,
            x="GDP per capita",
            y="Life_expectancy",
            animation_frame="year",
            animation_group="country",
            size="Population",
            color="GDP_group",
            hover_name="country",
            hover_data={
                "Population size": True,
                "Life_expectancy": ":.1f",
                "GDP per capita": ":,.0f",
                "Population": False
            },
            log_x=True,
            size_max=60,
            title="",
            color_discrete_map={
                "Faible revenu": "#e74c3c",
                "Intermédiaire": "#f39c12",
                "Haut revenu": "#2ecc71"
            }
        )
    else:
        df_year = df_plot[df_plot['year'] == selected_year_display]
        fig1 = px.scatter(
            df_year,
            x="GDP per capita",
            y="Life_expectancy",
            size="Population",
            color="GDP_group",
            hover_name="country",
            hover_data={
                "Population size": True,
                "Life_expectancy": ":.1f",
                "GDP per capita": ":,.0f",
                "Population": False
            },
            log_x=True,
            size_max=60,
            title=f"Année {selected_year_display}",
            color_discrete_map={
                "Faible revenu": "#e74c3c",
                "Intermédiaire": "#f39c12",
                "Haut revenu": "#2ecc71"
            }
        )
    
    fig1.update_layout(
        xaxis_title="PIB par habitant $ (échelle logarithmique)",
        yaxis_title="Espérance de vie à la naissance (années)",
        legend_title="Niveau de développement",
        height=600,
        hovermode='closest'
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Insights
    with st.expander("💡 Insights clés"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_low = df_plot[df_plot['GDP_group'] == 'Faible revenu']['Life_expectancy'].mean()
            st.metric("🔴 Faible revenu", f"{avg_low:.1f} ans")
        
        with col2:
            avg_mid = df_plot[df_plot['GDP_group'] == 'Intermédiaire']['Life_expectancy'].mean()
            st.metric("🟡 Intermédiaire", f"{avg_mid:.1f} ans")
        
        with col3:
            avg_high = df_plot[df_plot['GDP_group'] == 'Haut revenu']['Life_expectancy'].mean()
            st.metric("🟢 Haut revenu", f"{avg_high:.1f} ans")
        
        st.write(f"📊 **Écart total** : {avg_high - avg_low:.1f} ans entre pays riches et pauvres")

# ============================================================================
# TAB 4 : MODÈLE DE RÉGRESSION
# ============================================================================
with tab4:
    st.header("📈 Modèle de Régression : Résidus")
    
    st.subheader("📊 Écart de chaque pays par rapport à la tendance PIB ↔️ longévité")
    
    # Préparation des données
    df_plot_resid = filtered_df.copy()
    
    # S'assurer que GDP_group existe
    if 'GDP_group' not in df_plot_resid.columns:
        df_plot_resid = df_plot_resid.dropna(subset=["Life_expectancy", "GDP per capita"])
        df_plot_resid = df_plot_resid[df_plot_resid["GDP per capita"] > 0]
        df_plot_resid["GDP_group"] = df_plot_resid.groupby("year")["GDP per capita"].transform(
            lambda x: pd.qcut(x, q=3, labels=["Faible revenu", "Intermédiaire", "Haut revenu"], duplicates='drop')
        )
    
    # Calcul des résidus
    residus = []
    for year, group in df_plot_resid.groupby("year"):
        group = group.dropna(subset=["Life_expectancy", "GDP per capita"])
        group = group[group["GDP per capita"] > 0]
        
        if len(group) < 10:
            continue
        
        X = np.log(group["GDP per capita"])
        X = sm.add_constant(X)
        y = group["Life_expectancy"]
        
        try:
            model = sm.OLS(y, X).fit()
            
            residus.append(pd.DataFrame({
                "country": group["country"],
                "year": year,
                "Residual_LE": (y - model.predict(X)).round(2),
                "GDP per capita": group["GDP per capita"],
                "Population": group["Population"],
                "GDP_group": group["GDP_group"]
            }))
        except:
            continue
    
    if residus:
        df_residual = pd.concat(residus, ignore_index=True)
        
        # Formatage
        df_residual["Population_hover"] = df_residual["Population"].apply(
            lambda x: f"{int(x):,}".replace(",", " ") if pd.notna(x) else "N/A"
        )
        
        # Options d'affichage
        col_a, col_b = st.columns([3, 1])
        
        with col_b:
            show_animation_resid = st.checkbox("Animation temporelle", value=True, key="anim_resid")
            selected_year_resid = st.slider(
                "Année affichée",
                min_value=int(df_residual['year'].min()),
                max_value=int(df_residual['year'].max()),
                value=int(df_residual['year'].max()),
                key="year_resid"
            )
        
        # Graphique
        if show_animation_resid:
            fig4 = px.scatter(
                df_residual,
                x="GDP per capita",
                y="Residual_LE",
                animation_frame="year",
                animation_group="country",
                size="Population",
                color="Residual_LE",
                hover_name="country",
                hover_data={
                    "Residual_LE": ":.2f",
                    "Population_hover": True,
                    "GDP per capita": ":,.0f",
                    "Population": False
                },
                color_continuous_scale="RdYlGn",
                log_x=True,
                size_max=60,
                title=""
            )
        else:
            df_resid_year = df_residual[df_residual['year'] == selected_year_resid]
            fig4 = px.scatter(
                df_resid_year,
                x="GDP per capita",
                y="Residual_LE",
                size="Population",
                color="Residual_LE",
                hover_name="country",
                hover_data={
                    "Residual_LE": ":.2f",
                    "Population_hover": True,
                    "GDP per capita": ":,.0f",
                    "Population": False
                },
                color_continuous_scale="RdYlGn",
                log_x=True,
                size_max=60,
                title=f"Année {selected_year_resid}"
            )
        
        fig4.update_layout(
            xaxis_title="PIB par habitant (échelle logarithmique)",
            yaxis_title="Résidu espérance de vie (années)",
            coloraxis_colorbar_title="Résidu (années)",
            height=600
        )
        
        st.plotly_chart(fig4, use_container_width=True)
        
        with st.expander("💡 Interprétation des résidus"):
            st.markdown("""
            Les **résidus** montrent l'écart entre l'espérance de vie observée et celle prédite par le modèle PIB :
            - **Résidu positif (vert)** : Le pays a une espérance de vie supérieure à ce que prédit son PIB
            - **Résidu négatif (rouge)** : Le pays a une espérance de vie inférieure à ce que prédit son PIB
            - **Résidu proche de 0** : Le PIB explique bien l'espérance de vie
            """)
            
            # Top 5 positifs et négatifs
            latest_year = df_residual['year'].max()
            df_latest = df_residual[df_residual['year'] == latest_year]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🟢 Top 5 sur-performants**")
                top5_pos = df_latest.nlargest(5, 'Residual_LE')[['country', 'Residual_LE']]
                st.dataframe(top5_pos, hide_index=True)
            
            with col2:
                st.markdown("**🔴 Top 5 sous-performants**")
                top5_neg = df_latest.nsmallest(5, 'Residual_LE')[['country', 'Residual_LE']]
                st.dataframe(top5_neg, hide_index=True)
    else:
        st.warning("⚠️ Pas assez de données pour calculer les résidus avec les filtres actuels")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("**✨ Dashboard créé avec Streamlit** | Données : Espérance de vie mondiale")
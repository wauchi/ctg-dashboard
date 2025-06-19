import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from joblib import load

st.set_page_config(
    page_title="Human Modification Index",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_model():
    return load("dashboard/model/Gradient_Boosting_Model.pkl")
    # return 0


@st.cache_resource
def load_df():
    return pd.read_csv("dashboard/data/Modell_Features.csv")


def numeric_input(label, value, min_val=None, max_val=None, step=0.1):
    if isinstance(value, int):
        return st.number_input(
            label, value=value, step=1, min_value=min_val, max_value=max_val
        )
    else:
        return st.slider(
            label,
            min_value=min_val or 0.0,
            max_value=max_val or 100.0,
            value=float(value),
            step=step,
        )


model = load_model()
df = load_df()


st.title("Human Modification Index")

with st.sidebar:
    with st.expander("Quellen:"):
        st.markdown(
            """
            * [Demokratische Mitsprache ist wichtig – auch beim Verbandsbeschwerderecht. (o. J.)](https://www.chgemeinden.ch/de/newsroom/beitrag/2023_07_05_Demokratische-Mitsprache-ist-wichtig-auch-beim-Verbandsbeschwerderecht.php)
            * [Die Schweiz wird städtischer: Die Bevölkerungsentwicklung im Laufe der Zeit (o. J.)](https://www.bfs.admin.ch/asset/de/31005128)
            * [Wie es tatsächlich um die «Stadtflucht» und den «Babyboom» steht. Holzer Thomas. (o. J.).](https://staedteverband.ch/de/detail/wie-es-tatsachlich-um-die-laquo-stadtflucht-raquo-und-den-laquo-babyboom-raquo-steht?share=1)
            * [Vom Stadtleben ins Landleben: Der Trend zur Stadtflucht. ZDFheute. ](https://www.zdfheute.de/wirtschaft/stadtflucht-deutschland-land-leben-kodorf-100.html)
            * [Städte werden zu teuer für Mittelstand. Blick. ](https://www.blick.ch/wirtschaft/mittelstand-kann-sich-die-hohen-mieten-nicht-mehr-leisten-arme-staedte-sie-werden-zu-reichen-ghettos-id18388514.html)
            * [Mietpreise in der Schweiz 2024: Der stärkste Anstieg seit 20 Jahren. ](https://www.mietkautionschweiz.ch/blog/mietpreise-in-der-schweiz-2024)
            * [Suburbanisierung: Ursachen und Auswirkungen der Stadtflucht. Utopia.de.](https://utopia.de/ratgeber/suburbanisierung-ursachen-und-auswirkungen-der-stadtflucht_230713/)
            * [Nachhaltige Städte und Gemeinden. (o. J.).](https://www.agenda-2030.eda.admin.ch/de/sdg-11-nachhaltige-staedte-und-gemeinden)
            * [Suburbanisierung. (2025). In Wikipedia. ](https://de.wikipedia.org/w/index.php?title=Suburbanisierung&oldid=253686260))
                    """
        )

st.markdown(
    """
**​Herzlich willkommen zum Human Modification Index Dashboard!​**

Dieses interaktive Dashboard bietet die Möglichkeit mit vordefinierten Szenarien – "Stadtflucht", "Landflucht" und "Alternde Gesselschaft" – und freien Erwartungen Prognosen für zukünftige Entwicklungen erstellen. Nutzen Sie die verfügbaren Visualisierungen und interaktiven Elemente, um verschiedene Annahmen zu vergleichen und fundierte Einblicke zu gewinnen.​
Das Dashboard wurde im Rahmen des CTG1-Moduls an der FHNW in Brugg entwickelt.

Viel Spass beim Erkunden!
Adrian, Raphael und Nils
"""
)

st.caption(
    """Der Human Modification Index (GHM) ist ein Mass dafür, wie stark menschliche Aktivitäten natürliche Landschaften verändert haben. Er hilft dabei, den Einfluss des Menschen auf verschiedene Ökosysteme weltweit zu verstehen und zu quantifizieren. Ein hoher HMI-Wert weist auf eine intensive menschliche Nutzung hin, wie sie in Städten oder industriellen Gebieten vorkommt. Ein niedriger Wert deutet auf weitgehend natürliche und wenig beeinträchtigte Gebiete hin. 
Der Index basiert auf Analysen von 13 verschiedenen menschlichen Einflüsse, wie zum Beispiel die Bevölkerungsdichte, die Verkehrsinfrastruktur oder die Energiegewinnung. _[ESSD](https://essd.copernicus.org/articles/12/1953/2020/)_
"""
)

st.image(
    Image.open("dashboard/img/map.png"), caption="Darstellung GHM in der Schweiz (2015)"
)
st.markdown(
    "Die Karte offenbart eine enorme Bandbreite menschlicher Einflussnahme im Schweizer Landschaftsraum. Während die rötlich markierten Agglomerationen des Mittellands auf stark umgestaltete, dicht besiedelte Gebiete hinweisen, zeigen sich im hochalpinen Süden und in den inneralpinen Tälern tiefblaue Zonen mit minimaler Modifikation. Ein deutlicher Nord-Süd-Gradient sticht hervor: Vom industrialisierten Flachland über die Voralpen bis hin zu weitgehend unbebauten Gebirgsregionen sinkt der Index stetig ab. Auffällig ist zudem, dass selbst innerhalb einzelner Kantone kleinräumig starke Kontraste zwischen intensiv genutzten Talböden und naturnahen Höhenlagen vorkommen. Insgesamt illustriert die Visualisierung, wie dicht bebautes Territorium und unzugängliche Wildnis in der Schweiz auf engstem Raum nebeneinanderliegen."
)


df_gemeinde = pd.read_csv("dashboard/data/parsed_gemeinde.csv")
selected_gemeinden = [
    "Pully",
    "Davos",
    "Hasliberg",
    "Aarau",
]
df_gemeinde = df_gemeinde[df_gemeinde["Gemeindename"].isin(selected_gemeinden)]

fig, ax = plt.subplots(figsize=(6, 4))

infos = {
    "Pully": ("blue", "Pully (Stadt am Genfersee)"),
    "Davos": ("green", "Davos (Dorf mit grossem Berggebiet)"),
    "Wyssachen": ("red", "Hasliberg (Dorf mit viel Natur- und Landwirtschaftsfläche)"),
    "Aarau": ("orange", "Aarau (Kleine Stadt)"),
}


for gemeinde in df_gemeinde["Gemeindename"].unique():
    data_subset = df_gemeinde[df_gemeinde["Gemeindename"] == gemeinde]
    info = infos.get(gemeinde, ("black", gemeinde))
    ax.plot(
        data_subset["jahr"],
        data_subset["wert"],
        marker="o",
        linewidth=2,
        label=info[1],
        color=info[0],
    )

ax.set_xlabel("Jahr")
ax.set_ylabel("Wert")
ax.set_title("Zeitliche Entwicklung der Werte pro Gemeinde")

ax.legend(title="Gemeinde", bbox_to_anchor=(1.05, 1), loc="upper left")
ax.grid(True)

st.pyplot(fig)

st.markdown(
    "Seit der ersten Erhebung 1990 zeigt der Global Human Modification Index in sämtlichen Teilen der Schweiz einen klaren Aufwärtstrend. Sowohl in städtischen Zentren als auch in abgelegenen Alpentälern nehmen die Werte kontinuierlich zu, getrieben von Infrastruktur­ausbau, Tourismus und verdichteter Besiedlung. Damit rückt selbst bisher wenig verändertes Terrain sukzessive in höhere Einflussklassen auf."
)

st.divider()

st.markdown(
    "Nachfolgend können Vorhersagen für drei realistische Szenarien getätigt werden. Im freien Szenarion können Merkmale ohne Einschränkungen verändert und individuelle Szenraien aufgezeigt werden."
)
tabs_list = [
    "⛰️ Stadtflucht",
    "🏙️ Landflucht",
    "👵 Alternde Gesellschaft",
    "🔓 Freies Szenario",
]
s1, s2, s3, sf = st.tabs(tabs_list)

with s1:
    st.markdown(
        """Wir wollten untersuchen, welche Auswirkungen eine Stadtflucht auf den GHM hat. Dieses Phänomen, bei dem Städte mehr Abwanderung als Neuzuzüge verzeichnen, ist seit einigen Jahren vor allem in Deutschland zu beobachten (Hufnagel Sarah, 2024). Auch in der Schweiz ist dies seit der Coronapandemie ein viel diskutiertes Thema. Allerdings ist aktuell keine enorme Stadtflucht zu verzeichnen, sodass unsere Modellrechnung sicherlich übertrieben ist (Holzer Thomas, o. J.). Wichtig zu beachten ist, dass der Begriff „Stadtflucht” bzw. „Suburbanisierung” nicht dasselbe wie eine „Entstädterung” ist. Bei einer Stadtflucht handelt es sich um die Flucht vom Stadtzentrum in das nähere und weitere Umland. Bei einer Entstädterung wandert die Bevölkerung aus der Stadt und dem nahen Umland komplett aufs Land („Suburbanisierung“, 2025)."""
    )

    st.markdown(
        """In der Schweiz, wo wir unsere Analyse durchführen, sind die Begriffe verschwommener. Die Schweiz ist sehr dicht besiedelt. Deshalb ist es schwierig, von einer klaren Entstädterung zu sprechen, da man beispielsweise auch bei einem Umzug von Bern in ein Dorf wie Herzogenbuchsee nach wie vor innerhalb von 30 Minuten im Stadtzentrum von Bern ist, obwohl Herzogenbuchsee nicht zur Agglomeration von Bern gehört. Wir unterscheiden deshalb in unserem Modell auch nicht."""
    )

    st.markdown(
        """Für unser Modell verwenden wir die vom BFS als statistische Städte definierten Gemeinden. Von diesen wird jeweils 30 % der Bevölkerung abgezogen. Dieser Teil wird der Bevölkerung der anderen Gemeinden anschliessend gleichmässig hinzugefügt. Dies ist natürlich nicht vollständig realistisch, da kleine Gemeinden einen genauso grossen Zuzug erhalten wie grössere Gemeinden. Trotzdem ist es aktuell schwierig zu sagen, wie eine Suburbanisierung eines so grossen Teils effektiv aussehen würde, weshalb wir diese Annahme treffen."""
    )

    img = Image.open("dashboard/img/s1.png")
    st.image(img, caption="Stadtflucht")

    st.markdown(
        """Der GHM nimmt im Schnitt um etwa 0,02 zu. Auffallend ist vor allem, dass der Schnitt in den Städten leicht abnimmt, in den Dörfern aber stärker ansteigt. Dies ist nachvollziehbar, da die Bevölkerung in den Städten abnimmt, während ein Grossteil der Bebauung etc. gleich bleibt. Auf dem Land werden hingegen neue Wohnmöglichkeiten geschaffen und vor allem in kleineren Dörfern steigt die Bevölkerungsdichte teilweise massiv an."""
    )

    st.markdown(
        """Wenn Menschen aus der Stadt ins Umland ziehen, verändert sich nicht nur die Bevölkerungsverteilung, sondern auch die Landnutzung und die Umwelt. In unserem Modell ist zu sehen, dass der GHM-Wert in kleinen Gemeinden ansteigt. Das bedeutet, dass der Einfluss des Menschen auf die Umwelt zunimmt, beispielsweise durch neue Wohngebiete, Strassen oder Infrastruktur.
Dabei haben verschiedene Gruppen unterschiedliche Interessen:
*	Zuziehende suchen günstigen Wohnraum, Natur und Lebensqualität.
*	Einheimische fürchten oft steigende Bodenpreise, den Verlust von Grünflächen oder Veränderungen in der Dorfgemeinschaft.
*	Politik und Bauwirtschaft haben wiederum eigene Ziele, zum Beispiel Wachstum oder Gewinn.
*	Städte haben Angst, Firmen zu verlieren oder durch mehr Pendler ein Verkehrschaos zu erleben.

Der Zugang zu Land (also: Wer darf wo bauen?) ist ungleich verteilt. Reiche Haushalte, Immobilienfirmen oder einflussreiche Gemeinden können besser mitentscheiden. Andere Menschen oder Gruppen, wie etwa junge Familien oder ältere Dorfbewohner, verlieren oft ihr Mitspracherecht.
Stadtflucht ist somit nicht nur ein geografischer Prozess, sondern auch ein politischer und sozialer Konflikt, bei dem es um Zugang, Kontrolle und Gerechtigkeit geht. Wer diesen Wandel gestaltet – und wer nicht –, hat direkte Auswirkungen auf Umwelt und Gesellschaft. Gerade wenn es um Infrastruktur und Arbeitsplätze geht, gibt es viele Fragen: Was muss wo gebaut werden und wer zahlt dafür? (Rau, 2021)
"""
    )

with s2:
    st.markdown(
        """Wie bereits im ersten Szenario wollen wir auch in diesem einen Umzug der Bevölkerung analysieren. Dabei geht es nun aber um das Gegenteil, wie verhält sich der GHM bei einem Wegzug vom Land zur Stadt. Dieses Szenario ist in der Schweiz eigentlich bereits eingetreten aber über einen langen Zeitraum. Vor Hundert Jahren war 1/3 der Bevölkerung in Städten während heutzutage 3/4 in der Stadt Leben(Die Schweiz wird städtischer, o. J.). Wir wollen schauen, wie es aussehen würde, wenn sich dieser Trend fortsetzt."""
    )
    st.markdown(
        """Wir verwenden das gleiche Modell wie bei der Stadtflucht, kehren diesmal aber das Szenario um, heisst 30% jedes Dorfes wird entfernt und gleichmässig auf die Städte verteilt. Auch hier wieder der Hinweis, dass wir bei der gleichmässigen Verteilung der Zuzüger eine grobe Annahme getätigt haben."""
    )

    img = Image.open("dashboard/img/s2.png")
    st.image(img, caption="Landflucht")

    st.markdown(
        """Der gesamte GHM sinkt ganz leicht um etwa 0.015 aber Wert in den Städten steigt um etwa 0.02 und sinkt in den Dörfern um 0.02. Auch dies macht Sinn, da die Bevölkerungsdichte in den Städten zunimmt und in den Dörfern abnimmt."""
    )
    st.markdown(
        """Aus Sicht der Politischen Ökologie ist auch der Wegzug vom Land in die Stadt – wie ihn unser Modell darstellt – mehr als nur eine demografische Bewegung. Es geht um ungleiche Verteilungen von Raum, Ressourcen und Entscheidungsmacht, die durch solche Wanderungsprozesse verstärkt oder verändert werden."""
    )

    st.markdown(
        """Wenn mehr Menschen in die Städte ziehen, verändert sich dort nicht nur die Bevölkerungsdichte, sondern auch der Zugang zu Wohnraum, Infrastruktur und Boden. In der Realität führt dies oft zu steigenden Mietpreisen und Verdrängung einkommensschwächerer Bevölkerungsteile – ein Prozess, den man auch Gentrifizierung nennt (Mietkautionschweiz.ch, o. J.).Gewinner sind meist Investoren und einkommensstarke Haushalte, während ärmere Menschen – oder auch Migrant*innen – an den Stadtrand oder ganz aus dem urbanen Raum gedrängt werden."""
    )
    st.markdown(
        """In ländlichen Regionen bedeutet der Bevölkerungsrückgang hingegen oft: Weniger öffentliche Investitionen, Schliessung von Schulen oder Läden und Verlust politischer Aufmerksamkeit. So verlieren diese Regionen an sozialer und politischer Macht. Auch hier stellt die Politische Ökologie die Frage: Wer entscheidet eigentlich, was mit diesen Gebieten geschieht – und auf wessen Kosten?"""
    )
    st.markdown(
        """Narrative wie „Urbanisierung ist Fortschritt“ oder „Stadtleben ist nachhaltig“ dienen dabei oft als strategische Werkzeuge, um bestimmte politische oder wirtschaftliche Interessen durchzusetzen. Auch wenn diese Argumente teilweise stimmen, blenden sie soziale Ungleichheiten aus – etwa wer sich das Stadtleben leisten kann und wer nicht (Kolbe, 2023)."""
    )
with s3:
    st.markdown(
        """Im Rahmen dieses Szenarios wurde untersucht, wie sich eine hypothetische gesellschaftliche Entwicklung, konkret eine alternde Bevölkerung (Zunahme von Personen über 65 Jahren und Abnahme von Personen unter 65 Jahren), eine sinkende Geburtenrate (- 30%) und politische Verschiebungen (Zunahme der Parteien, welche von Personen über 65 Jahre vermehrt gewählt werden) – auf den Global Human Modification Index (GHM) auf Gemeindeebene in der Schweiz auswirkt. Dabei sank der durchschnittliche GHM-Wert von 0.63 auf etwa 0.59. """
    )

    img = Image.open("dashboard/img/s3.png")
    st.image(img, caption="Alternde Gesellschaft")

    st.markdown(
        """Eine alternde Gesellschaft bringt veränderte politische und gesellschaftliche Prioritäten mit sich. Ältere Menschen haben häufig ein höheres Sicherheits- und Erhaltungsbedürfnis und legen verstärkt Wert auf Ruhe, Stabilität und Bewahrung des Bestehenden – insbesondere im Hinblick auf ihre Wohnumgebung und Mobilität (Ruehli, o. J.). Gleichzeitig treten sie seltener als treibende Kraft für neue Umweltprojekte oder ökologische Transformationen auf. Jüngere Generationen hingegen setzen andere Schwerpunkte: Klimaschutz, energieeffiziente Wohnmodelle oder eine nachhaltige Raumplanung.
Daraus ergeben sich konkurrierende Interessen: Sollen freie Flächen für neue Pflegeinfrastruktur oder doch für urbane Lebensräume, Schulen oder Grünzonen genutzt werden? Wie viel Umbau der Städte und Gemeinden ist gesellschaftlich gewünscht – und von wem?
"""
    )

    st.markdown(
        """Die Schweiz hat eine besonders hohe Eigentumsquote unter älteren Menschen (Alter, 2024). In Kombination mit deren wachsender politischer Bedeutung entstehen neue Machtverhältnisse in der Raum- und Umweltpolitik. Viele Ältere sind Immobilieneigentümer:innen und setzen sich für Stabilität, Eigentumsschutz und restriktive Bauvorschriften ein. Sie profitieren vom Werterhalt und der Begrenzung neuer Bauprojekte in ihrer Umgebung – auch wenn diese aus ökologischer oder sozialer Sicht sinnvoll wären. So werden junge Familien oder einkommensschwächere Gruppen strukturell benachteiligt: Sie haben weniger Zugang zu günstigem Wohnraum und weniger Mitspracherecht bei kommunalen Entwicklungsprojekten (Zehn Jahre RPG 1, 2024)."""
    )

    st.markdown(
        """
Die Interessen der älteren Bevölkerung werden durch vielfältige Strategien gestützt:
*	Narrative wie „Schutz des Ortsbilds“, „Tradition bewahren“ oder „Heimat schützen“ wirken politisch stabilisierend.
*	Organisationen wie konservative Parteien oder Eigentümerverbände vertreten gezielt deren Anliegen.
*	Regularien wie restriktive Zonenpläne hemmen Transformation und fördern die Erhaltung des Status quo (Zehn Jahre RPG 1, 2024).

Diese Strategien wirken auf vielen Ebenen – und führen dazu, dass Nachhaltigkeits- oder Klimaziele lokal schwerer durchzusetzen sind, selbst wenn sie national oder international formuliert sind.
"""
    )

    st.markdown(
        """Der Rückgang des GHM-Werts in diesem Szenario mag auf den ersten Blick positiv erscheinen – als Zeichen für weniger Flächenverbrauch oder geringere Infrastrukturbelastung. Doch genauer betrachtet ist er kein Ausdruck aktiver Umweltpolitik, sondern eine Folge demografisch und politisch bedingter Entwicklung: weniger Bautätigkeit, abnehmende Bevölkerung unter 65, geringerer Innovationsdruck."""
    )

with sf:
    st.markdown(
        "Hier können nun eigene Simulationen durchgeführt werden. Dazu kann eine Gemeinde im Dropdown ausgewählt werden und anschliessend die Paramter bearbeitet werden. \n\n Falls Simulationen für eine frei gefunden Gemeinde durchgeführt werden sollen, kann im Dropdown `Freie Gemeinde` ausgewählt werden."
    )

    gemeinde = st.selectbox("Wähle eine Gemeinde", df["Gemeindename"].unique())
    row = df[df["Gemeindename"] == gemeinde].iloc[0]

    einwohner = st.number_input(
        "Einwohner  (2019)", 0.0, 420000.0, row["Einwohner  (2019)"]
    )
    veraenderung = st.slider(
        "Veränderung in %  (2010-2019)",
        -50.0,
        50.0,
        row["Veränderung in %  (2010-2019)"],
    )
    dichte = st.number_input(
        "Bevölkerungsdichte pro km²  (2019)",
        0.0,
        13000.0,
        row["Bevölkerungs-dichte pro km²  (2019)"],
    )
    auslaender = st.slider(
        "Ausländer in %  (2019)", 0.0, 60.0, row["Ausländer in %  (2019)"]
    )
    jugend = st.slider("0-19 Jahre  (2019)", 0.0, 100.0, row["0-19 Jahre  (2019)"])
    erwachsen = st.slider("20-64 Jahre  (2019)", 0.0, 100.0, row["0-19 Jahre  (2019)"])
    senioren = st.slider(
        "65 Jahre und mehr  (2019)", 0.0, 100.0, row["65 Jahre und mehr  (2019)"]
    )
    heirat = st.number_input(
        "Rohe Heiratssziffer  (2019)", 0.0, 37.0, row["Rohe Heiratssziffer  (2019)"]
    )
    scheidung = st.number_input(
        "Rohe Scheidungsziffer  (2019)", 0.0, 39.0, row["Rohe Scheidungsziffer  (2019)"]
    )
    geburten = st.number_input(
        "Rohe Geburtenziffer  (2019)", 0.0, 47.0, row["Rohe Geburtenziffer  (2019)"]
    )
    sterbe = st.number_input(
        "Rohe Sterbeziffer  (2019)", 0.0, 58.0, row["Rohe Sterbeziffer  (2019)"]
    )
    haushalte = st.number_input(
        "Anzahl Privathaushalte  (2019)",
        0.0,
        205000.0,
        row["Anzahl Privathaushalte  (2019)"],
    )
    haushaltsgr = st.number_input(
        "Durchschnittliche Haushaltsgrösse in Personen  (2019)",
        0.0,
        5.0,
        row["Durchschnittliche Haushaltsgrösse in Personen  (2019)"],
    )
    leerstand = st.number_input(
        "Leerwohnungsziffer  (2020)",
        0.0,
        14.0,
        row["Leerwohnungs-ziffer  (2020)"],
    )
    neubau = st.number_input(
        "Neu gebaute Wohnungen pro 1000 Einwohner  (2018)",
        0.0,
        97.0,
        row["Neu gebaute Wohnungen pro 1000 Einwohner  (2018)"],
    )

    fdp = st.slider("FDP  (2019)", 0.0, 100.0, float(row["FDP  (2019)"]))
    cvp = st.slider("CVP  (2019)", 0.0, 100.0, float(row["CVP  (2019)"]))
    sp = st.slider("SP  (2019)", 0.0, 100.0, float(row["SP  (2019)"]))
    svp = st.slider("SVP  (2019)", 0.0, 100.0, float(row["SVP  (2019)"]))
    evp = st.slider("EVP/CSP  (2019)", 0.0, 100.0, float(row["EVP/CSP  (2019)"]))
    glp = st.slider("GLP  (2019)", 0.0, 100.0, float(row["GLP  (2019)"]))
    bdp = st.slider("BDP  (2019)", 0.0, 100.0, float(row["BDP  (2019)"]))
    pda = st.slider("PdA/Sol.  (2019)", 0.0, 100.0, float(row["PdA/Sol.  (2019)"]))
    gps = st.slider("GPS  (2019)", 0.0, 100.0, float(row["GPS  (2019)"]))
    kleinrechts = st.slider(
        "Kleine Rechtsparteien  (2019)",
        0.0,
        100.0,
        float(row["Kleine Rechtsparteien  (2019)"]),
    )

    # Liste aller Werte
    parteiwerte = [fdp, cvp, sp, svp, evp, glp, bdp, pda, gps, kleinrechts]
    parteien = [
        "FDP",
        "CVP",
        "SP",
        "SVP",
        "EVP/CSP",
        "GLP",
        "BDP",
        "PdA/Sol.",
        "GPS",
        "Kleine Rechtsparteien",
    ]

    # Gesamtsumme berechnen
    gesamt = sum(parteiwerte)

    # Skalierung durchführen (nur wenn Summe > 0)
    if gesamt > 0:
        skaliert = [(w / gesamt) * 100 for w in parteiwerte]
        # Optionale Anzeige
        st.markdown("**Skalierte Parteianteile (Summe = 100 %)**")
        for name, wert in zip(parteien, skaliert):
            st.write(f"{name}: {wert:.2f} %")
    else:
        st.warning("Summe der Parteianteile ist 0 %. Bitte Werte setzen.")
        skaliert = [0.0] * len(parteien)

    # Sonderfall: Berggebiet (bool)
    berggebiet = st.selectbox(
        "Berggebiet", ["Ja", "Nein"], index=0 if row["Berggebiet"] else 1
    )

    if st.button("Vorhersage starten"):
        try:
            input_vector = [
                einwohner,
                veraenderung,
                dichte,
                auslaender,
                jugend,
                erwachsen,
                senioren,
                heirat,
                scheidung,
                geburten,
                sterbe,
                haushalte,
                haushaltsgr,
                leerstand,
                neubau,
                fdp,
                cvp,
                sp,
                svp,
                evp,
                glp,
                bdp,
                pda,
                gps,
                kleinrechts,
                1 if berggebiet == "Ja" else 0,
            ]

            columns = [
                "Einwohner  (2019)",
                "Veränderung in %  (2010-2019)",
                "Bevölkerungs-dichte pro km²  (2019)",
                "Ausländer in %  (2019)",
                "0-19 Jahre  (2019)",
                "20-64 Jahre  (2019)",
                "65 Jahre und mehr  (2019)",
                "Rohe Heiratssziffer  (2019)",
                "Rohe Scheidungsziffer  (2019)",
                "Rohe Geburtenziffer  (2019)",
                "Rohe Sterbeziffer  (2019)",
                "Anzahl Privathaushalte  (2019)",
                "Durchschnittliche Haushaltsgrösse in Personen  (2019)",
                "Leerwohnungs-ziffer  (2020)",
                "Neu gebaute Wohnungen pro 1000 Einwohner  (2018)",
                "FDP  (2019)",
                "CVP  (2019)",
                "SP  (2019)",
                "SVP  (2019)",
                "EVP/CSP  (2019)",
                "GLP  (2019)",
                "BDP  (2019)",
                "PdA/Sol.  (2019)",
                "GPS  (2019)",
                "Kleine Rechtsparteien  (2019)",
                "Berggebiet",
            ]

            input_df = pd.DataFrame([input_vector], columns=columns)

            prognose = model.predict(input_df)
            value = row["GHM_Mean_2020"]

            st.success(f"Aktueller Wert: {value:.4f}")
            st.success(f"Vorhersage: {prognose[0]:.4f}")
        except Exception as e:
            st.error(f"Fehler bei der Vorhersage: {e}")

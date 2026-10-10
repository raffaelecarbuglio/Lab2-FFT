"""Build a compact, three-page PDF from the generated numerical results."""

from pathlib import Path
import csv
from matplotlib.font_manager import FontProperties, findfont
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parent


def main():
    for name, weight in (("Sans", "normal"), ("SansBold", "bold")):
        pdfmetrics.registerFont(TTFont(name, findfont(FontProperties(family="DejaVu Sans", weight=weight))))
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="SansBold", italic="Sans", boldItalic="SansBold")
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "SansBold" if "Heading" in style.name or style.name == "Title" else "Sans"
    styles.add(ParagraphStyle("Body", fontName="Sans", fontSize=9.5,
                              leading=14, spaceAfter=9))
    styles.add(ParagraphStyle("Caption", fontName="Sans", fontSize=8, leading=11, alignment=TA_CENTER,
                              textColor=colors.HexColor("#45556a"), spaceAfter=10))
    story = []

    def text(value, style="Body"):
        story.append(Paragraph(value, styles[style]))

    def figure(name, width, height, caption):
        story.append(Image(str(ROOT / "results" / name), width=width, height=height))
        text(caption, "Caption")

    text("Laboratory 2 | Fourier analysis", "Title")
    text("Reproducible reconstruction of the University of Pisa FFT exercise", "Caption")
    text("<b>Data provenance.</b> All records and figures are synthetic. Original Arduino "
         "measurements were not provided. This report gives the theoretical expectations "
         "and demonstrates the analysis; it does not report experimental measurements.")
    text("1. Periodic waveforms", "Heading2")
    text("With 2048 samples separated by 100 microseconds, the sampling frequency is "
         "10,000 Hz, the Nyquist frequency is 5,000 Hz, and the bin spacing is "
         "4.8828125 Hz. The fundamental at 78.125 Hz completes exactly 16 cycles "
         "in the 0.2048 s DFT record, avoiding partial-cycle leakage.")
    figure("waveforms.png", 465, 362, "Unit-amplitude synthetic signals and their one-sided FFT amplitudes.")
    text("A sine gives one spectral line. Square and triangular waves have odd "
         "harmonics, with relative amplitudes 1/h and 1/h<super>2</super>. "
         "The triangle's smoother waveform needs less high-frequency content.")
    rows = [["h", "Square FFT", "Theory", "Triangle FFT", "Theory"]]
    with (ROOT / "results/harmonics.csv").open() as stream:
        harmonics = list(csv.DictReader(stream))
    for h in (3, 5, 7, 9):
        square = next(r for r in harmonics if r["waveform"] == "square" and int(r["harmonic"]) == h)
        triangle = next(r for r in harmonics if r["waveform"] == "triangle" and int(r["harmonic"]) == h)
        rows.append([str(h), f'{float(square["fft_ratio"]):.6f}', f"{1/h:.6f}",
                     f'{float(triangle["fft_ratio"]):.6f}', f"{1/h**2:.6f}"])
    table = Table(rows, colWidths=[35, 107, 107, 107, 107])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                               ("FONTNAME", (0, 0), (-1, -1), "Sans"),
                               ("FONTNAME", (0, 0), (-1, 0), "SansBold"),
                               ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                               ("FONTSIZE", (0, 0), (-1, -1), 9),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.append(table)

    story.append(PageBreak())
    text("2. Damped oscillator", "Heading1")
    text("We use x(t) = exp(-gamma t) sin(omega_d t) for t &gt;= 0, where "
         "omega_d<super>2</super> = omega_0<super>2</super> - gamma<super>2</super>. "
         "The natural frequency is fixed at 781.25 Hz; gamma is either 50 or 150 s<super>-1</super>.")
    text("Its continuous transform is omega_d / [omega_0<super>2</super> - "
         "(2 pi f)<super>2</super> + 4 pi i gamma f]. This follows by integrating "
         "the two decaying complex exponentials that make up the sine.")
    figure("damping.png", 465, 181, "FFT points approximate the continuous impulse-response transform (lines).")
    text("<b>Interpretation.</b> Stronger damping shortens the train of oscillations. "
         "Fewer coherent cycles give a broader frequency peak. With the same initial "
         "amplitude, the integrated response is smaller, so its peak is lower.")
    text("Near resonance, for weak damping, the power spectrum has full width at "
         "half maximum approximately gamma/pi: 15.9155 and 47.7465 Hz. The modulus "
         "has a different half-height width, approximately sqrt(3) gamma/pi. "
         "The quality factor is omega_0/(2 gamma).")
    with (ROOT / "results/damping.csv").open() as stream:
        damping = list(csv.DictReader(stream))
    for row in damping:
        text(f'gamma = {float(row["gamma_per_s"]):g} s<super>-1</super>: '
             f'FFT peak bin {float(row["fft_peak_bin_hz"]):.4f} Hz; continuous peak '
             f'{float(row["continuous_peak_hz"]):.4f} Hz; maximum complex FFT error '
             f'relative to the exact discrete reference {float(row["max_relative_dft_error"]):.2e}.')
    text("The finite sampled transform is independently evaluated using geometric sums. "
         "The continuum peak is sqrt(omega_0<super>2</super> - 2 gamma<super>2</super>)/(2 pi), "
         "slightly different from the ringdown frequency. The 4.8828125 Hz FFT grid "
         "cannot show that small shift directly.")
    text("<b>Connection to RLC.</b> The transform of an impulse response is proportional "
         "to the corresponding transfer function. This comparison assumes the appropriate "
         "initial conditions and observed variable. Capacitor and resistor voltages, or "
         "ringdowns with a different initial phase, need not have the same numerator.")
    text("For the transient we plot dt times |FFT|, in signal units times seconds, "
         "without mean subtraction or windowing. This is different from the periodic "
         "waveforms' single-sided peak amplitudes.")

    story.append(PageBreak())
    text("3. Finite records and experimental data", "Heading1")
    text("A pure sine completing 16.4 cycles does not join smoothly across the ends "
         "of the record. Its spectral weight spreads over neighbouring bins: this is "
         "spectral leakage. A Hann window reduces distant tails while broadening the "
         "central peak. Broadening alone therefore does not prove physical distortion.")
    figure("leakage.png", 465, 174, "The same synthetic off-bin sine with and without a Hann window.")
    text("<b>Normalization.</b> For an unwindowed record of N samples, the one-sided "
         "amplitude is 2|FFT|/N at interior positive frequencies. DC and, for even N, "
         "Nyquist are not doubled. With Hann weighting the denominator is the sum of "
         "the weights. This corrects the amplitude of an isolated bin-centred tone, "
         "but not the peak-bin amplitude of every off-bin tone.")
    text("<b>Sampling.</b> Frequencies above Nyquist fold into the observed band "
         "(aliasing). Ideal square and triangular waves are not band-limited, which "
         "explains small sampling differences in the harmonic table. NumPy does not "
         "require a power-of-two record length. Zero padding adds plotting points "
         "rather than new frequency information.")
    text("<b>Real recordings.</b> analyze.py reads two-column time/signal records "
         "and converts microseconds, milliseconds, or seconds. It checks strictly "
         "increasing finite timestamps. Deviations above 1% of the mean sample "
         "interval require explicit opt-in. This threshold is only a practical "
         "warning: accepting the mean interval does not correct timing jitter.")
    text("The DFT duration is N dt; the first-to-last timestamp interval is (N-1)dt. "
         "A larger sampling rate extends the observable band; a longer observation "
         "improves frequency separation. Bin spacing is not a statistical error bar.")
    text("<b>Reproduce.</b> Run demo.py, build_report.py, and "
         "python -m unittest discover -s tests -v. See README.md for installation "
         "and recording-analysis commands; solution.md contains the full explanation.")
    story.append(Spacer(1, 8))
    text('Source exercise: <link href="https://osiris.df.unipi.it/~fuso/dida/fft_2016.pdf">'
         'Francesco Fuso, FFT exercises, University of Pisa (12 April 2017)</link>.', "Caption")
    text('Implementation references: <link href="https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html">'
         'NumPy rfft</link> and <link href="https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftfreq.html">'
         'rfftfreq</link>.', "Caption")

    def footer(canvas, doc):
        canvas.setFont("Sans", 8)
        canvas.setFillColor(colors.HexColor("#617083"))
        canvas.drawString(42, 25, "Lab2-FFT | Synthetic reconstruction")
        canvas.drawRightString(553, 25, str(doc.page))

    document = SimpleDocTemplate(str(ROOT / "report.pdf"), pagesize=(595.28, 841.89),
                                 rightMargin=42, leftMargin=42, topMargin=36, bottomMargin=42,
                                 title="Laboratory 2 - Fourier analysis", author="Lab2-FFT")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print("Generated report.pdf")


if __name__ == "__main__":
    main()

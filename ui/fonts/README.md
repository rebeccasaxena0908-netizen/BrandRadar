Fonts (SIL Open Font License 1.1, see OFL-*.txt and https://openfontlicense.org):

- `Montserrat-SemiBold.ttf`: latin subset from Fontsource (https://fontsource.org/fonts/montserrat), used by
  `wordcloud`, which needs a font file rather than CSS. Montserrat by Julieta Ulanovsky et al.
- `Montserrat-Bold.ttf`: full static font from https://github.com/JulietaUla/Montserrat (fonts/ttf).
- `Figtree-Regular.ttf`, `Figtree-SemiBold.ttf`, `Figtree-Bold.ttf`: static fonts from
  https://github.com/erikdkennedy/figtree (fonts/ttf). Figtree by Erik Kennedy.

The PDF report (reports/weekly.py) embeds these in every PDF, and reports/render.py embeds them in the
chart images, so reports look the same on any machine.

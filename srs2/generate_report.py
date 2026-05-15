from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

doc = Document()

# ─── Поля страницы ───
section = doc.sections[0]
section.top_margin = Cm(2.5)
section.bottom_margin = Cm(2.5)
section.left_margin = Cm(3)
section.right_margin = Cm(1.5)

# ─── Стили ───
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(12)

def set_font(run, bold=False, size=12, color=None, italic=False):
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)

def add_heading(doc, text, level=1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if level == 1 else WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    set_font(run, bold=True, size=16 if level == 1 else (14 if level == 2 else 13),
             color=(0x1F, 0x49, 0x7D))
    p.space_before = Pt(12)
    p.space_after = Pt(6)
    return p

def add_paragraph(doc, text, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = align
    run = p.add_run(text)
    set_font(run)
    p.space_after = Pt(6)
    p.paragraph_format.first_line_indent = Cm(1.25)
    return p

def add_bullet(doc, text):
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(text)
    set_font(run)
    p.space_after = Pt(3)
    return p

def add_code(doc, code_text):
    for line in code_text.strip().split('\n'):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(1)
        run = p.add_run(line if line else ' ')
        run.font.name = 'Courier New'
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x00, 0x00, 0x80)
        p.space_after = Pt(0)
        # Серый фон
        shading = OxmlElement('w:shd')
        shading.set(qn('w:val'), 'clear')
        shading.set(qn('w:color'), 'auto')
        shading.set(qn('w:fill'), 'F2F2F2')
        p._p.get_or_add_pPr().append(shading)

def add_separator(doc):
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), '1F497D')
    pBdr.append(bottom)
    pPr.append(pBdr)
    p.space_after = Pt(6)

# ══════════════════════════════════════════════
# ТИТУЛЬНАЯ СТРАНИЦА
# ══════════════════════════════════════════════
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('Министерство образования и науки\nРеспублики Казахстан')
set_font(run, size=12)

doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('ОТЧЁТ\nпо мини-проекту')
set_font(run, bold=True, size=18, color=(0x1F, 0x49, 0x7D))

doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('Классификация сельскохозяйственных культур\nс использованием спутниковых снимков\nи машинного обучения в Google Earth Engine')
set_font(run, bold=True, size=14)

doc.add_paragraph()
doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('Дисциплина: Введение в интеллектуальные системы управления')
set_font(run, size=12, italic=True)

doc.add_paragraph()
doc.add_paragraph()
doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
run = p.add_run(f'Дата выполнения: {datetime.date.today().strftime("%d.%m.%Y")}')
set_font(run, size=12)

doc.add_page_break()

# ══════════════════════════════════════════════
# СОДЕРЖАНИЕ
# ══════════════════════════════════════════════
add_heading(doc, 'СОДЕРЖАНИЕ', 1)
contents = [
    ('1.', 'Введение'),
    ('2.', 'Цель работы'),
    ('3.', 'Используемые инструменты и данные'),
    ('4.', 'Описание исходных данных (AOI)'),
    ('5.', 'Получение спутниковых снимков Sentinel-2 и Sentinel-1'),
    ('6.', 'Описание модели EU Crop Map 2022'),
    ('7.', 'Стадия 1: Классификация на 7 major классов'),
    ('8.', 'Стадия 2: Классификация на 19 minor классов'),
    ('9.', 'Экспорт данных'),
    ('10.', '3D визуализация в Kepler.gl'),
    ('11.', 'Результаты и выводы'),
]
for num, title in contents:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1)
    run = p.add_run(f'{num}  {title}')
    set_font(run, size=12)
    p.space_after = Pt(3)

doc.add_page_break()

# ══════════════════════════════════════════════
# 1. ВВЕДЕНИЕ
# ══════════════════════════════════════════════
add_heading(doc, '1. Введение', 2)
add_separator(doc)
add_paragraph(doc,
    'Дистанционное зондирование Земли (ДЗЗ) является одним из наиболее эффективных методов '
    'мониторинга сельскохозяйственных угодий. Современные спутниковые системы, такие как '
    'Sentinel-2 Европейского космического агентства (ESA), позволяют получать снимки земной '
    'поверхности с разрешением 10 метров на пиксель каждые 5 дней, что обеспечивает возможность '
    'регулярного наблюдения за состоянием посевов.'
)
add_paragraph(doc,
    'В данной работе применяется платформа Google Earth Engine (GEE) — облачная геоинформационная '
    'система, позволяющая обрабатывать петабайты спутниковых данных без необходимости локальной '
    'установки специализированного ПО. Совместное использование GEE и алгоритмов машинного '
    'обучения (в частности, Random Forest) открывает широкие возможности для автоматической '
    'идентификации типов сельскохозяйственных культур.'
)

# ══════════════════════════════════════════════
# 2. ЦЕЛЬ
# ══════════════════════════════════════════════
add_heading(doc, '2. Цель работы', 2)
add_separator(doc)
add_paragraph(doc,
    'Целью данного мини-проекта является разработка и применение системы автоматической '
    'классификации сельскохозяйственных культур на основе мультиспектральных спутниковых '
    'снимков Sentinel-2 и Sentinel-1 с использованием алгоритма Random Forest в платформе '
    'Google Earth Engine, а также визуализация полученных результатов в виде трёхмерной карты.'
)
p = doc.add_paragraph()
run = p.add_run('Задачи:')
set_font(run, bold=True)
tasks = [
    'Загрузить и подготовить зону интереса (AOI) в формате Shapefile.',
    'Получить и предобработать мультиспектральные снимки Sentinel-2 и Sentinel-1 за 2022 год.',
    'Применить двухуровневую классификацию Random Forest (7 major → 19 minor классов).',
    'Определить типы сельскохозяйственных культур в пределах исследуемого полигона.',
    'Экспортировать результаты классификации и построить 3D карту в Kepler.gl.',
]
for t in tasks:
    add_bullet(doc, t)

# ══════════════════════════════════════════════
# 3. ИНСТРУМЕНТЫ
# ══════════════════════════════════════════════
add_heading(doc, '3. Используемые инструменты и данные', 2)
add_separator(doc)

table = doc.add_table(rows=1, cols=2)
table.style = 'Table Grid'
table.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr = table.rows[0].cells
hdr[0].text = 'Инструмент / Данные'
hdr[1].text = 'Описание'
for cell in hdr:
    run = cell.paragraphs[0].runs[0]
    set_font(run, bold=True, size=11)
    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

rows_data = [
    ('Google Earth Engine (GEE)', 'Облачная платформа для обработки спутниковых данных'),
    ('Sentinel-2 SR Harmonized', 'Мультиспектральный спутник ESA, разрешение 10м/пиксель'),
    ('Sentinel-1 GRD', 'Радарный спутник ESA (SAR), поляризация VV/VH'),
    ('ASTER DEM', 'Цифровая модель рельефа'),
    ('EU Crop Map 2022', 'Предобученная модель классификации культур (BOKU, Вена)'),
    ('Kepler.gl', 'Платформа для 3D геовизуализации'),
    ('Shapefile (test1)', 'Зона интереса — полигон исследуемого поля'),
]
for r1, r2 in rows_data:
    row = table.add_row().cells
    row[0].text = r1
    row[1].text = r2
    for cell in row:
        set_font(cell.paragraphs[0].runs[0] if cell.paragraphs[0].runs else cell.paragraphs[0].add_run(''), size=11)

doc.add_paragraph()

# ══════════════════════════════════════════════
# 4. ИСХОДНЫЕ ДАННЫЕ
# ══════════════════════════════════════════════
add_heading(doc, '4. Описание исходных данных (AOI)', 2)
add_separator(doc)
add_paragraph(doc,
    'Зона интереса (Area of Interest, AOI) предоставлена в формате ESRI Shapefile и содержит '
    'один полигон сельскохозяйственного поля, расположенного в Центральной Азии (Казахстан). '
    'Shapefile состоит из следующих файлов:'
)
files = [
    ('.shp', 'геометрия полигона (координаты вершин)'),
    ('.shx', 'индекс для быстрого доступа к геометрии'),
    ('.dbf', 'атрибутивная таблица (поле Id = 0)'),
    ('.prj', 'система координат: WGS 1984 UTM Zone 41S'),
    ('.cpg', 'кодировка текста'),
]
for ext, desc in files:
    add_bullet(doc, f'{ext} — {desc}')

add_paragraph(doc,
    'Координаты ограничивающего прямоугольника (bounding box) в метрах: '
    'X: 1 125 475 – 1 126 909, Y: 15 681 040 – 15 682 848. '
    'Площадь поля составляет приблизительно 1.4 км × 1.8 км. '
    'При загрузке в GEE файлы .sbn и .sbx были исключены, так как они являются '
    'пространственными индексами ESRI и не принимаются платформой.'
)

p = doc.add_paragraph()
run = p.add_run('Код загрузки AOI в GEE:')
set_font(run, bold=True)

add_code(doc, """// Загрузка зоны интереса из Assets
var aoi = ee.FeatureCollection('projects/srs1-488806/assets/test1_srs2');
var geometry = aoi.geometry();

// Отображение полигона на карте
Map.centerObject(aoi, 13);
Map.addLayer(aoi, {color: 'red'}, 'AOI');""")

# ══════════════════════════════════════════════
# 5. СПУТНИКОВЫЕ ДАННЫЕ
# ══════════════════════════════════════════════
add_heading(doc, '5. Получение спутниковых снимков Sentinel-2 и Sentinel-1', 2)
add_separator(doc)
add_paragraph(doc,
    'Для классификации использовались снимки двух спутниковых миссий за 2022 год. '
    'Sentinel-2 предоставляет мультиспектральные данные в 10 каналах с пространственным '
    'разрешением 10 м/пиксель. Sentinel-1 предоставляет радарные данные (SAR) в режиме IW '
    'с поляризацией VV и VH, что позволяет анализировать структуру посевов независимо '
    'от облачности.'
)
add_paragraph(doc,
    'Предобработка Sentinel-2 включала: фильтрацию по облачности (< 50%), маскирование '
    'облаков по вероятности (> 75%) и SCL, вычисление вегетационных индексов (NDVI, SAVI, '
    'EVI2, LAI, MSAVI, BSI, NDWI, MNDWI, NDTI, NDBI и др.). '
    'Предобработка Sentinel-1 включала: спекл-фильтр (фокусированная медиана, радиус 30м), '
    'извлечение признаков из VV и VH каналов. '
    'Для каждого из 8 месяцев (март–октябрь) вычислялись медианные месячные композиты, '
    'а также годовые перцентильные статистики (5-й, 50-й, 98-й перцентили).'
)

p = doc.add_paragraph()
run = p.add_run('Код предобработки данных:')
set_font(run, bold=True)

add_code(doc, """var year = '2022';
var date_start = ee.Date(year + '-01-01');
var date_end   = ee.Date(year + '-12-31');
var bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];

// ─── Sentinel-2 ───
var cloudprob = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
  .filterDate(date_start, date_end).filterBounds(geometry);

var dataSE = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate(date_start, date_end)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
  .filterBounds(geometry);

// Join S2 с вероятностью облачности
var idJoin    = ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'});
var innerJoin = ee.Join.saveFirst('Cloud', 'system:time_start', true, null);
dataSE = ee.ImageCollection(innerJoin.apply(dataSE, cloudprob, idJoin));

// Предобработка и вегетационные индексы
dataSE = preproces_S2(dataSE, 75).select(bands);
var Collection_S2_filtered = vegetation_index(dataSE);

// Месячные и годовые признаки S2
var months    = ee.List.sequence(3, 10);
var S2_Monthly = S2month(months, Collection_S2_filtered, year);
var S2_Yearly  = ee.Image(Collection_S2_filtered
  .reduce(ee.Reducer.percentile([5, 50, 98])));
S2_Yearly = S2_Yearly.rename(newNames_func(S2_Yearly, year));

// ─── Sentinel-1 ───
var radar = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filterBounds(geometry)
  .filterDate(year+'-01-01', year+'-12-31')
  .filter(ee.Filter.eq('instrumentMode','IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
  .select('VV','VH');
radar = preproces_S1(radar, geometry);
radar = feat_s1(radar);
var S1_Monthly = S1month(radar, year+'-01-01', 12, 1, 'month');
var S1_Yearly  = radar.reduce(ee.Reducer.percentile([5, 50, 98]));

// ─── DEM ───
var dem = ee.Image("projects/sat-io/open-datasets/ASTER/GDEM")
  .clip(geometry.bounds());

// ─── Стек всех данных ───
var data = S2_Monthly.toBands()
  .addBands(S2_Yearly)
  .addBands(S1_Monthly.toBands())
  .addBands(S1_Yearly)
  .addBands(dem);""")

# ══════════════════════════════════════════════
# 6. МОДЕЛЬ EU CROP MAP
# ══════════════════════════════════════════════
add_heading(doc, '6. Описание модели EU Crop Map 2022', 2)
add_separator(doc)
add_paragraph(doc,
    'В работе использована модель EU Crop Map 2022, разработанная учёными Университета '
    'природных ресурсов и наук о жизни (BOKU, Вена, Австрия): '
    'Babak Ghassemi, Emma Izquierdo-Verdiguier, Francesco Vuolo. '
    'Модель предназначена для классификации сельскохозяйственных культур Европы на основе '
    'данных Sentinel-1, Sentinel-2 и MODIS.'
)
add_paragraph(doc,
    'Тренировочные данные содержат 21 708 точек, разделённых на 4 части '
    '(train_p1 ... train_p4), доступных через GEE Assets. '
    'Набор признаков (feat_select) содержит отобранные важные признаки, '
    'хранящиеся в FeatureCollection LU22_S1S2MYA_important_features2. '
    'Для нашего полигона из полного набора признаков модели было отобрано '
    '152 признака, доступных в данных за 2022 год.'
)

# ══════════════════════════════════════════════
# 7. СТАДИЯ 1
# ══════════════════════════════════════════════
add_heading(doc, '7. Стадия 1: Классификация на 7 major классов', 2)
add_separator(doc)
add_paragraph(doc,
    'На первой стадии Random Forest классифицирует каждый пиксель территории на один '
    'из 7 основных классов землепользования. Модель обучается на всех тренировочных точках '
    'с меткой Lbl_cls_major. Параметры Random Forest: 130 деревьев, минимальный размер '
    'листа = 2, доля бутстрэп-выборки = 0.5.'
)

p = doc.add_paragraph()
run = p.add_run('7 основных классов:')
set_font(run, bold=True)

classes7 = [
    ('0', 'Artificial land', 'Искусственные территории (города, дороги)'),
    ('1', 'Arable land', 'Пахотные земли (поля под культуры)'),
    ('2', 'Woodland & Shrubland', 'Леса и кустарники'),
    ('4', 'Grassland', 'Луга и пастбища'),
    ('5', 'Bare land', 'Голые земли, пустоши'),
    ('6', 'Water', 'Водные объекты'),
    ('7', 'Wetlands', 'Водно-болотные угодья'),
]
table = doc.add_table(rows=1, cols=3)
table.style = 'Table Grid'
hdr = table.rows[0].cells
for i, h in enumerate(['Класс', 'Название (EN)', 'Описание']):
    hdr[i].text = h
    set_font(hdr[i].paragraphs[0].runs[0], bold=True, size=11)
    hdr[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

for c, en, ru in classes7:
    row = table.add_row().cells
    row[0].text = c
    row[1].text = en
    row[2].text = ru
    for cell in row:
        if cell.paragraphs[0].runs:
            set_font(cell.paragraphs[0].runs[0], size=11)

doc.add_paragraph()
add_paragraph(doc,
    'Результат для исследуемого полигона: поле определено как класс 1 — '
    'Arable land (пахотные земли), что является корректным результатом для '
    'сельскохозяйственного поля в Казахстане.'
)

p = doc.add_paragraph()
run = p.add_run('Код классификации (Стадия 1):')
set_font(run, bold=True)

add_code(doc, """// Отбор доступных признаков
var feat_select_all = f_name1.aggregate_array('0').distinct();
var available_bands = data.clip(geometry).bandNames();
var feat_select = feat_select_all
  .filter(ee.Filter.inList('item', available_bands));

// Заполнение пустых пикселей нулями
var final_data = data.select(feat_select).unmask(0).clip(geometry);

// Обучение Random Forest — 7 major классов
var model = ee.Classifier.smileRandomForest({
  numberOfTrees: 130,
  minLeafPopulation: 2,
  bagFraction: 0.5,
  seed: 0
});
model = model.train(train_p1, 'Lbl_cls_major', feat_select);
model = model.train(train_p2, 'Lbl_cls_major', feat_select);
model = model.train(train_p3, 'Lbl_cls_major', feat_select);
model = model.train(train_p4, 'Lbl_cls_major', feat_select);

// Классификация
var class_map_major = final_data.classify(model);

// Визуализация
Map.centerObject(geometry, 14);
Map.addLayer(Visualization(class_map_major), {}, 'Культуры (7 классов)');
Add_legend();""")

# ══════════════════════════════════════════════
# 8. СТАДИЯ 2
# ══════════════════════════════════════════════
add_heading(doc, '8. Стадия 2: Классификация на 19 minor классов', 2)
add_separator(doc)
add_paragraph(doc,
    'На второй стадии модель углубляется внутрь класса Arable land и определяет конкретный '
    'тип культуры из 19 возможных. Тренировочные точки фильтруются только по пахотным землям '
    '(Lbl_cls_major = 1), целевая метка — Label_clas. '
    'Параметры Random Forest: 150 деревьев (больше для точности), минимальный размер листа = 2.'
)

p = doc.add_paragraph()
run = p.add_run('19 minor классов культур:')
set_font(run, bold=True)

classes19 = [
    ('1', 'Common wheat', 'Мягкая пшеница'),
    ('2', 'Durum wheat', 'Твёрдая пшеница'),
    ('3', 'Barley', 'Ячмень'),
    ('4', 'Rye', 'Рожь'),
    ('5', 'Oats', 'Овёс'),
    ('6', 'Maize', 'Кукуруза'),
    ('7', 'Rice', 'Рис'),
    ('8', 'Triticale', 'Тритикале'),
    ('9', 'Other cereals', 'Прочие зерновые'),
    ('10', 'Potatoes', 'Картофель'),
    ('11', 'Sugar beet', 'Сахарная свёкла'),
    ('12', 'Other root crops', 'Прочие корнеплоды'),
    ('13', 'Non-perm. industrial crops', 'Непостоянные технические культуры'),
    ('14', 'Sunflower', 'Подсолнечник'),
    ('15', 'Rape and turnip rape', 'Рапс'),
    ('16', 'Textile crops', 'Текстильные культуры (хлопок)'),
    ('17', 'Dry pulses', 'Сухие бобовые'),
    ('18', 'Vegetables', 'Овощи'),
    ('19', 'Other/Fallow', 'Прочие / Паровые'),
]
table = doc.add_table(rows=1, cols=3)
table.style = 'Table Grid'
hdr = table.rows[0].cells
for i, h in enumerate(['Класс', 'Название (EN)', 'Описание']):
    hdr[i].text = h
    set_font(hdr[i].paragraphs[0].runs[0], bold=True, size=10)
    hdr[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

for c, en, ru in classes19:
    row = table.add_row().cells
    row[0].text = c
    row[1].text = en
    row[2].text = ru
    for cell in row:
        if cell.paragraphs[0].runs:
            set_font(cell.paragraphs[0].runs[0], size=10)

doc.add_paragraph()

p = doc.add_paragraph()
run = p.add_run('Код классификации (Стадия 2):')
set_font(run, bold=True)

add_code(doc, """// Обучение Random Forest — 19 minor классов
// Фильтруем только пахотные земли (Lbl_cls_major = 1)
var model_minor = ee.Classifier.smileRandomForest({
  numberOfTrees: 150,
  minLeafPopulation: 2,
  bagFraction: 0.5,
  seed: 0
});
model_minor = model_minor.train(
  train_p1.filter(ee.Filter.eq('Lbl_cls_major', 1)),
  'Label_clas', feat_select);
model_minor = model_minor.train(
  train_p2.filter(ee.Filter.eq('Lbl_cls_major', 1)),
  'Label_clas', feat_select);
model_minor = model_minor.train(
  train_p3.filter(ee.Filter.eq('Lbl_cls_major', 1)),
  'Label_clas', feat_select);
model_minor = model_minor.train(
  train_p4.filter(ee.Filter.eq('Lbl_cls_major', 1)),
  'Label_clas', feat_select);

// Классификация на 19 классов
var class_map_minor = final_data.classify(model_minor);

// Проверка результатов
print('Значения классов:',
  class_map_minor.sample({region: geometry, scale: 10, numPixels: 10})
  .aggregate_array('classification'));

// Визуализация с легендой
Map.addLayer(Visualization(class_map_minor), {}, 'Культуры (19 классов)');
Add_legend();""")

add_paragraph(doc,
    'Результат классификации для исследуемого полигона: модель определила два типа культур. '
    'Класс 6 (Maize — Кукуруза) составляет основную часть поля, класс 1 (Common Wheat — '
    'Мягкая пшеница) присутствует на отдельных участках. Значения классов, '
    'полученные при выборке 10 пикселей: [1, 6, 6, 6, 6, 6, 6, 6, 6, 6].'
)

# ══════════════════════════════════════════════
# 9. ЭКСПОРТ
# ══════════════════════════════════════════════
add_heading(doc, '9. Экспорт данных', 2)
add_separator(doc)
add_paragraph(doc,
    'После классификации все пиксели полигона экспортированы в формате CSV через Google Drive. '
    'Файл содержит координаты каждой точки и присвоенный класс культуры. '
    'Всего экспортировано 20 673 точки — по одной на каждый пиксель 10×10 м внутри полигона.'
)

p = doc.add_paragraph()
run = p.add_run('Структура экспортированного CSV файла:')
set_font(run, bold=True)

cols = [
    ('system:index', 'Уникальный идентификатор точки'),
    ('classification', 'Класс культуры (1 = пшеница, 6 = кукуруза)'),
    ('.geo', 'GeoJSON с координатами точки (lon, lat)'),
]
table = doc.add_table(rows=1, cols=2)
table.style = 'Table Grid'
hdr = table.rows[0].cells
for i, h in enumerate(['Колонка', 'Описание']):
    hdr[i].text = h
    set_font(hdr[i].paragraphs[0].runs[0], bold=True, size=11)
for c, d in cols:
    row = table.add_row().cells
    row[0].text = c
    row[1].text = d
    for cell in row:
        if cell.paragraphs[0].runs:
            set_font(cell.paragraphs[0].runs[0], size=11)

doc.add_paragraph()

p = doc.add_paragraph()
run = p.add_run('Код экспорта:')
set_font(run, bold=True)

add_code(doc, """// Экспорт всех пикселей полигона без ограничения количества
var crop_points = class_map_minor.sample({
  region: geometry,
  scale: 10,          // разрешение 10м x 10м
  geometries: true    // включаем координаты каждой точки
});

Export.table.toDrive({
  collection: crop_points,
  description: 'crop_19classes',
  fileFormat: 'CSV'
});""")

# ══════════════════════════════════════════════
# 10. ВИЗУАЛИЗАЦИЯ
# ══════════════════════════════════════════════
add_heading(doc, '10. 3D визуализация в Kepler.gl', 2)
add_separator(doc)
add_paragraph(doc,
    'Для трёхмерной визуализации результатов классификации использована платформа Kepler.gl — '
    'мощный инструмент геовизуализации с открытым исходным кодом, разработанный компанией Uber. '
    'Платформа работает непосредственно в браузере без установки дополнительного ПО.'
)
add_paragraph(doc,
    'Экспортированный CSV файл (20 673 точки) был загружен в Kepler.gl. '
    'Каждая точка отображается по географическим координатам (lon/lat), '
    'цвет присваивается на основе колонки classification по схеме Custom Ordinal: '
    'класс 1 (пшеница) — зелёный цвет, класс 6 (кукуруза) — жёлтый цвет. '
    'Трёхмерный эффект достигается наклоном карты с помощью правой кнопки мыши.'
)

p = doc.add_paragraph()
run = p.add_run('Настройки визуализации в Kepler.gl:')
set_font(run, bold=True)
settings = [
    'Тип слоя: Point',
    'Color Based On: classification',
    'Color Scale: Custom Ordinal',
    'Класс 1 (Common Wheat): зелёный (#1a9641)',
    'Класс 6 (Maize): жёлтый (#fdae61)',
    '3D режим: правая кнопка мыши + перетаскивание',
]
for s in settings:
    add_bullet(doc, s)

# ══════════════════════════════════════════════
# 11. РЕЗУЛЬТАТЫ
# ══════════════════════════════════════════════
add_heading(doc, '11. Результаты и выводы', 2)
add_separator(doc)

p = doc.add_paragraph()
run = p.add_run('Основные результаты:')
set_font(run, bold=True)

results = [
    'Полигон успешно загружен в GEE и использован как зона интереса.',
    'Получены и предобработаны снимки Sentinel-2 и Sentinel-1 за 2022 год (152 признака).',
    'Двухуровневая классификация Random Forest выполнена успешно.',
    'Стадия 1 определила поле как Arable land (пахотные земли).',
    'Стадия 2 определила два типа культур: Кукуруза (класс 6) — основная культура поля, Пшеница (класс 1) — на отдельных участках.',
    'Экспортировано 20 673 точки (все пиксели полигона при разрешении 10м×10м).',
    'Построена интерактивная 3D карта в Kepler.gl.',
]
for r in results:
    add_bullet(doc, r)

doc.add_paragraph()

p = doc.add_paragraph()
run = p.add_run('Выводы:')
set_font(run, bold=True)

add_paragraph(doc,
    'В ходе выполнения мини-проекта была успешно реализована система автоматической '
    'классификации сельскохозяйственных культур на основе спутниковых данных. '
    'Двухуровневый подход (7 major → 19 minor классов) позволил сначала определить '
    'общий тип землепользования, а затем идентифицировать конкретные культуры: '
    'кукурузу и пшеницу.'
)
add_paragraph(doc,
    'Использование Google Earth Engine обеспечило обработку большого объёма '
    'спутниковых данных в облаке без необходимости локальных вычислительных ресурсов. '
    'Алгоритм Random Forest показал высокую эффективность при работе с '
    'мультиспектральными и радарными данными.'
)
add_paragraph(doc,
    'Метод дистанционного зондирования с применением машинного обучения позволяет '
    'анализировать тысячи гектаров сельскохозяйственных угодий в автоматическом режиме, '
    'что имеет практическое значение для мониторинга посевов, оценки урожайности '
    'и планирования агрономических мероприятий.'
)

# Сохранение
path = '/Users/aboka/6B07113-ics-2026/srs2/report.docx'
doc.save(path)
print(f'Отчёт сохранён: {path}')

const base = location.pathname.startsWith('/platform') ? '/platform' : ''
const $ = id => document.getElementById(id)
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))
const date = value => value ? new Date(value).toLocaleString('zh-CN', {hour12:false}) : '—'
const COLORS = ['#38bdf8','#f59e0b','#a78bfa','#f87171','#4ade80','#22d3ee','#fb7185','#facc15']
const state = {
  meta:null, projects:[], project:null, images:[], image:null, boxes:[], selected:-1,
  datasets:[], models:[], jobs:[], filter:'all', job:null, timer:null, review:null, img:null,
  hoverBox:-1,
}

function toast(text, error=false){
  const el = $('toast')
  el.textContent = text
  el.className = 'show' + (error ? ' error' : '')
  clearTimeout(toast.timer)
  toast.timer = setTimeout(() => { el.className = '' }, 3500)
}

async function api(path, body, method){
  const options = {method: method || (body ? 'POST' : 'GET')}
  if (body instanceof FormData) options.body = body
  else if (body !== undefined){ options.body = JSON.stringify(body); options.headers = {'Content-Type':'application/json'} }
  const response = await fetch(base + path, options)
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw Error(data.error || ('请求失败 ' + response.status))
  return data
}

const labelOf = key => (state.meta?.classes || []).find(item => item.key === key)?.label || key
const colorOf = key => COLORS[Math.max(0, (state.meta?.det_classes || []).findIndex(item => item.key === key)) % COLORS.length]

function fillSelect(select, items, {empty='', value='key', label='label'} = {}){
  if (!select) return
  const previous = select.value
  select.replaceChildren()
  if (empty !== null) select.append(new Option(empty, ''))
  for (const item of items) select.append(new Option(item[label], item[value]))
  if (previous && [...select.options].some(option => option.value === previous)) select.value = previous
}

function switchView(view){
  if (['review','datasets','train'].includes(view)){
    view = 'home'
    if (location.hash !== '#home') history.replaceState(null, '', '#home')
  }
  document.querySelectorAll('.view').forEach(node => node.hidden = node.id !== 'view-' + view)
  document.querySelectorAll('#tabs button').forEach(button => button.classList.toggle('active', button.dataset.view === view))
  const hints = {
    home: '<b>先选右上角的产品/工位</b>，再按 2 步走：拍照/上传 → 框缺陷。框完保存即入库。',
    capture: '<b>第 1 步</b>：整块板放进画面拍清楚；同一批货填同一个批次。',
    label: '<b>第 2 步</b>：先在左上角选缺陷类型，再拖动鼠标圈出缺陷；画错了右键取消。',
    accuracy: '<b>每个缺陷的判断准确率</b>：数字低的类型可结合抽样结果检查。',
    datasets: '<b>训练数据</b>：把“检查通过”的图片做成训练数据，按板/批次分训练和测试。',
    train: '<b>训练模型</b>：选择训练数据，设置轮数后开始；页面会实时显示进度。',
    models: '<b>模型版本</b>：查看训练结果，发布后工位就会使用新模型。',
    accuracy: '<b>每个缺陷的判断准确率</b>：数字低的类型可结合抽样结果检查。',
    monitor: '<b>使用情况</b>：查看当前工位用的模型和现场反馈。',
    projects: '<b>先建产品/工位</b>：一个产品对应一个板型和工位。',
  }
  const banner = $('stepBanner')
  if (hints[view]){
    banner.innerHTML = hints[view]
    banner.hidden = false
  } else {
    banner.hidden = true
    banner.innerHTML = ''
  }
  refreshView(view)
}

function currentView(){
  return (location.hash || '#home').slice(1)
}

function needProject(){
  if (state.project) return true
  toast('请先选择或新建产品/工位', true)
  location.hash = 'projects'
  return false
}

async function refreshView(view){
  try{
    if (view === 'home'){ await loadHome() }
    else if (view === 'projects') await loadProjects()
    else if (view === 'capture'){ if (!needProject()) return; await loadProject(); await loadImages() }
    else if (view === 'label'){ if (!needProject()) return; await loadLabelImages() }
    else if (view === 'accuracy'){ if (!needProject()) return; await loadAccuracy() }
    else if (view === 'datasets'){ if (!needProject()) return; await loadDatasets() }
    else if (view === 'train'){ if (!needProject()) return; await loadDatasets(); await loadJobs() }
    else if (view === 'models'){ if (!needProject()) return; await loadModels() }
    else if (view === 'monitor'){ if (!needProject()) return; await loadMonitor() }
  } catch(error){ toast(error.message, true) }
}

/* ------------------------------------------------------------------ home */

function showHelp(){
  const dialog = $('helpDialog')
  if (!dialog.open) dialog.showModal()
  localStorage.setItem('pcbPlatformHelpSeen', '1')
}

async function loadHome(){
  const summaryNode = $('homeProjectSummary')
  if (!state.project){
    $('homeNew').textContent = '先新建产品/工位'
    $('homeLabeled').textContent = '—'
    $('homeReview').textContent = '—'
    summaryNode.innerHTML = '<p class="empty">还没有产品/工位。点右上角“新建产品/工位”，填产品名称和工位即可。</p>'
    return
  }
  const summary = await api('/api/projects/' + state.project.id + '/summary')
  $('homeNew').textContent = '共 ' + summary.total + ' 张'
  $('homeLabeled').textContent = summary.new + ' 张待框'
  $('homeReview').textContent = '点击查看'
  summaryNode.innerHTML = '<div class="panelhead"><h2>' + esc(state.project.name) + '</h2>'
    + '<span class="muted">' + esc(state.project.board_model || '未填板型') + ' · ' + esc(state.project.station || '未填工位') + '</span></div>'
    + '<div class="metric"><div><span>图片总数</span><strong>' + summary.total + '</strong></div>'
    + '<div><span>待框缺陷</span><strong>' + summary.new + '</strong></div>'
    + '<div><span>已保存</span><strong>' + summary.approved + '</strong></div>'
    + '<div><span>已驳回</span><strong>' + summary.rejected + '</strong></div>'
    + '<div><span>待处理反馈</span><strong>' + summary.feedback_open + '</strong></div></div>'
  await loadAccuracy()
}

/* --------------------------------------------------------------- accuracy */

async function loadAccuracy(){
  if (!state.project) return
  const data = await api('/api/projects/' + state.project.id + '/accuracy')
  renderAccuracy(data)
}

function renderAccuracy(data){
  const rows = (data.rows || []).filter(row => row.key !== 'good_solder')
  const source = data.source || '暂无已发布模型'
  $('accuracySource').textContent = source
  $('homeAccuracySource').textContent = source
  $('accuracyNote').textContent = data.note || ''
  const overall = data.overall || {}
  const pieces = []
  if (overall.top1 != null) pieces.push(['整体准确率', pct(overall.top1)])
  if (overall.map50 != null) pieces.push(['mAP50', pct(overall.map50)])
  if (overall.defect_precision != null) pieces.push(['缺陷精确率', pct(overall.defect_precision)])
  if (overall.defect_recall != null) pieces.push(['缺陷召回率', pct(overall.defect_recall)])
  $('accuracySummary').innerHTML = pieces.map(([label, value]) =>
    '<div><span>' + label + '</span><strong>' + value + '</strong></div>').join('')
    || '<div><span>指标</span><strong>—</strong></div>'
  const table = accuracyTable(rows)
  $('accuracyTable').innerHTML = table
  $('homeAccuracyTable').innerHTML = table
}

function accuracyTable(rows){
  if (!rows.length){
    return '<p class="empty">还没有可用的准确率数据。请先训练并发布模型。</p>'
  }
  return '<table><thead><tr><th>缺陷类型</th><th>缺陷召回率</th><th>报出后正确率</th><th>样本数</th></tr></thead><tbody>'
    + rows.map(row => {
      const accuracy = row.accuracy == null ? '未评估' : pct(row.accuracy)
      const precision = row.precision == null ? '—' : pct(row.precision)
      const klass = row.accuracy == null ? '' : row.accuracy >= .9 ? 'acc-good' : row.accuracy >= .7 ? 'acc-warn' : 'acc-bad'
      return '<tr><td><b>' + esc(row.label) + '</b></td>'
        + '<td><span class="acc ' + klass + '">' + accuracy + '</span></td>'
        + '<td>' + precision + '</td><td>' + (row.support == null ? '—' : row.support) + '</td></tr>'
    }).join('') + '</tbody></table>'
    + '<p class="muted" style="margin-top:8px">召回率 = 该类真实缺陷被检出的比例；样本数少时数字波动大，低于 80% 的类型可结合抽样结果检查。</p>'
}

/* ------------------------------------------------------------- projects */

async function loadProjects(){
  state.projects = await api('/api/projects')
  renderProjects()
  fillProjectSelect()
}

function renderProjects(){
  const list = $('projectList')
  list.replaceChildren()
  for (const project of state.projects){
    const card = document.createElement('button')
    card.className = 'card' + (state.project?.id === project.id ? ' active' : '')
    card.innerHTML = '<strong>' + esc(project.name) + '</strong>'
      + '<small>' + esc(project.board_model || '未填板型') + ' · ' + esc(project.station || '未填工位') + '</small>'
      + '<div class="meta"><span>图片 ' + project.images + '</span><span>待标注 ' + project.pending + '</span>'
      + '<span>数据集 ' + project.datasets + '</span><span>模型 ' + project.models + '</span></div>'
    card.onclick = () => selectProject(project.id).then(() => { location.hash = 'capture' })
    list.append(card)
  }
  if (!state.projects.length) list.innerHTML = '<p class="empty">还没有项目。下面的表单创建第一个。</p>'
  $('projectStats').textContent = state.projects.length + ' 个项目'
}

function fillProjectSelect(){
  const select = $('projectSelect')
  select.replaceChildren()
  if (!state.projects.length) select.append(new Option('暂无项目', ''))
  for (const project of state.projects) select.append(new Option(project.name + ' · ' + (project.station || '未填工位'), project.id))
  if (state.project && state.projects.some(project => project.id === state.project.id)) select.value = state.project.id
}

async function selectProject(id){
  if (!id) return
  state.project = await api('/api/projects/' + id)
  localStorage.setItem('pcbPlatformProject', id)
  fillProjectSelect()
  updateProjectLabels()
  if (currentView() !== 'projects') await refreshView(currentView())
}

function updateProjectLabels(){
  const name = state.project?.name || '未选择项目'
  $('captureTitle').textContent = '采集 / 上传 · ' + name
}

/* --------------------------------------------------------------- capture */

async function loadProject(){
  if (state.project) state.project = await api('/api/projects/' + state.project.id)
  const form = $('uploadForm')
  if (form){
    if (!form.elements.batch.value) form.elements.batch.value = localStorage.getItem('pcbPlatformBatch') || ''
    if (!form.elements.board_serial.value) form.elements.board_serial.value = localStorage.getItem('pcbPlatformBoard') || ''
  }
}

function fillStaticSelects(){
  fillSelect($('boxClass'), state.meta.det_classes, {empty:null})
  fillSelect($('datasetTemplate'), Object.entries(state.meta.templates).map(([key, value]) => ({key, label:value.label})), {empty:null})
}

async function loadImages(){
  const pid = state.project.id
  let query = ''
  if (state.filter === 'new') query = '?state=new'
  else if (state.filter === 'labeled') query = '?state=labeled'
  else if (state.filter !== 'all') query = '?review=' + state.filter
  state.images = await api('/api/projects/' + pid + '/images' + query)
  renderImages()
  $('imageCount').textContent = state.images.length + ' 张'
  $('captureStats').textContent = '图片 ' + state.project.images + ' · 已保存 ' + state.project.approved
  document.querySelectorAll('[data-img-filter]').forEach(button => button.classList.toggle('active', button.dataset.imgFilter === state.filter))
}

function renderImages(){
  const grid = $('imageGrid')
  grid.replaceChildren()
  for (const image of state.images){
    const card = document.createElement('article')
    card.className = 'image-card'
    const badge = image.state === 'new' ? 'new' : image.review_status
    const badgeText = image.state === 'new' ? '待框缺陷' : statusText(image.review_status)
    const open = document.createElement('button')
    open.type = 'button'
    open.className = 'open-training-image'
    open.innerHTML = '<img loading="lazy" src="' + base + '/api/images/' + image.id + '/file?w=320" alt="">'
      + '<span class="info"><strong title="' + esc(image.filename) + '">' + esc(image.filename) + '</strong>'
      + '<small>' + esc(image.batch || '无批次') + ' · ' + (image.label_count || 0) + ' 个框</small>'
      + '<span class="tags"><span class="badge ' + esc(badge) + '">' + badgeText + '</span></span></span>'
    open.onclick = async () => {
      try { await openImage(image.id); location.hash='label' }
      catch(error) { toast(error.message,true) }
    }
    const remove = document.createElement('button')
    remove.type = 'button'
    remove.className = 'danger'
    remove.textContent = '删除图片'
    remove.onclick = () => deleteTrainingImage(image.id)
    card.append(open, remove)
    grid.append(card)
  }
  if (!state.images.length) grid.innerHTML = '<p class="empty">还没有图片。先在上传区导入。</p>'
}

async function deleteTrainingImage(id) {
  try {
    const result = await api('/api/images/' + id, undefined, 'DELETE')
    if (state.image?.id === id) {
      state.image = null
      state.img = null
      state.boxes = []
      state.selected = -1
      $('labelCanvas').getContext('2d').clearRect(0,0,$('labelCanvas').width,$('labelCanvas').height)
      drawBoxList()
    }
    await loadProject()
    if (currentView() === 'label') await loadLabelImages()
    else await loadImages()
    toast(result.retained_in_versions ? '已从图库删除，历史数据集保留该图片快照' : '图片已删除')
  } catch(error) { toast(error.message, true) }
}
$('deleteTrainingImage').onclick = () => {
  if (state.image) deleteTrainingImage(state.image.id)
  else toast('请先选择图片',true)
}

function statusText(status){
  return {pending:'待处理', approved:'已保存', rejected:'已驳回', new:'待框缺陷'}[status] || status
}

/* --------------------------------------------------------------- labeling */

async function loadLabelImages(){
  state.images = await api('/api/projects/' + state.project.id + '/images?limit=300')
  const list = $('labelList')
  list.replaceChildren()
  for (const image of state.images){
    const row = document.createElement('div')
    row.className = 'row' + (state.image?.id === image.id ? ' active' : '')
    row.dataset.id = image.id
    row.innerHTML = '<img loading="lazy" src="' + base + '/api/images/' + image.id + '/file?w=160" alt="">'
      + '<div><strong>' + esc(image.filename) + '</strong><small>' + statusText(image.review_status) + ' · '
      + (image.label_count || 0) + ' 框</small></div>'
    row.onclick = () => openImage(image.id)
    list.append(row)
  }
  $('labelStats').textContent = state.images.length + ' 张图片'
  if (state.image && state.images.some(image => image.id === state.image.id)) openImage(state.image.id)
  else if (state.images.length) openImage(state.images[0].id)
  else $('canvasHint').textContent = '项目还没有图片，请先到“拍照/上传”上传。'
  updateLabelProgress()
}

function updateLabelProgress(){
  const index = state.images.findIndex(image => image.id === state.image?.id)
  $('imageProgress').textContent = (index >= 0 ? index + 1 : 0) + ' / ' + state.images.length
  $('prevImage').disabled = index <= 0
  $('nextImage').disabled = index < 0 || index >= state.images.length - 1
}

function moveImage(offset){
  const index = state.images.findIndex(image => image.id === state.image?.id)
  const next = index + offset
  if (next >= 0 && next < state.images.length) openImage(state.images[next].id)
}

$('prevImage').onclick = () => moveImage(-1)
$('nextImage').onclick = () => moveImage(1)

async function openImage(id){
  state.image = await api('/api/images/' + id)
  state.boxes = state.image.labels.map(item => ({class:item.class, x1:item.x1, y1:item.y1, x2:item.x2, y2:item.y2}))
  state.selected = -1
  state.hoverBox = -1
  hideHoverInfo()
  $('annotator').value = state.image.labels[0]?.annotator || localStorage.getItem('pcbPlatformAnnotator') || ''
  const image = new Image()
  image.onload = () => { if (state.image?.id === id) { state.img = image; drawCanvas() } }
  image.src = base + '/api/images/' + id + '/file?' + Date.now()
  document.querySelectorAll('#labelList .row').forEach(row => row.classList.toggle('active', row.dataset.id === id))
  drawBoxList()
  updateLabelProgress()
}

function boxAt(point){
  for (let index = state.boxes.length - 1; index >= 0; index--){
    const box = state.boxes[index]
    if (point[0] >= box.x1 && point[0] <= box.x2 && point[1] >= box.y1 && point[1] <= box.y2) return index
  }
  return -1
}

function showHoverInfo(){
  const info = $('hoverInfo')
  if (!info) return
  if (state.hoverBox < 0 || !state.boxes[state.hoverBox]){
    info.hidden = true
    return
  }
  const box = state.boxes[state.hoverBox]
  const canvas = $('labelCanvas'), wrap = canvas.parentElement
  const canvasRect = canvas.getBoundingClientRect(), wrapRect = wrap.getBoundingClientRect()
  info.textContent = '缺陷：' + labelOf(box.class)
  info.hidden = false
  const width = info.offsetWidth
  const left = canvasRect.left - wrapRect.left + box.x1 * canvasRect.width
  const top = canvasRect.top - wrapRect.top + box.y1 * canvasRect.height - 30
  info.style.left = Math.max(4, Math.min(left, wrapRect.width - width - 6)) + 'px'
  info.style.top = Math.max(4, top) + 'px'
}

function hideHoverInfo(){
  const info = $('hoverInfo')
  if (info) info.hidden = true
}

function drawCanvas(){
  const canvas = $('labelCanvas')
  if (!state.img) return
  const wrap = canvas.parentElement
  const maxWidth = Math.max(280, wrap.clientWidth - 24)
  const maxHeight = Math.max(300, Math.min(window.innerHeight - 300, 760))
  const ratio = Math.min(maxWidth / state.img.naturalWidth, maxHeight / state.img.naturalHeight, 2)
  const imageW = Math.max(1, Math.round(state.img.naturalWidth * ratio))
  const imageH = Math.max(1, Math.round(state.img.naturalHeight * ratio))
  canvas.width = imageW
  canvas.height = imageH
  const context = canvas.getContext('2d')
  context.clearRect(0, 0, imageW, imageH)
  context.drawImage(state.img, 0, 0, imageW, imageH)
  state.boxes.forEach((box, index) => {
    const color = colorOf(box.class)
    context.lineWidth = index === state.selected ? 3.5 : 2.4
    context.strokeStyle = color
    context.fillStyle = color + '22'
    const x = box.x1 * imageW, y = box.y1 * imageH
    const w = (box.x2 - box.x1) * imageW, h = (box.y2 - box.y1) * imageH
    context.fillRect(x, y, w, h)
    context.strokeRect(x, y, w, h)
  })
}

function canvasPoint(event){
  const canvas = $('labelCanvas')
  const rect = canvas.getBoundingClientRect()
  return [Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
          Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height))]
}

let dragStart = null
let previewBox = null
let pressBox = -1
let dragging = false

$('labelCanvas').addEventListener('pointerdown', event => {
  if (!state.img || event.button !== 0) return
  const point = canvasPoint(event)
  pressBox = boxAt(point)
  state.selected = -1
  dragging = false
  dragStart = point
  $('labelCanvas').setPointerCapture(event.pointerId)
})

$('labelCanvas').addEventListener('pointermove', event => {
  if (!dragStart){
    const boxIndex = boxAt(canvasPoint(event))
    if (boxIndex !== state.hoverBox){
      state.hoverBox = boxIndex
      showHoverInfo()
    }
    $('labelCanvas').style.cursor = boxIndex >= 0 ? 'pointer' : 'crosshair'
    return
  }
  const point = canvasPoint(event)
  const distance = Math.abs(point[0]-dragStart[0]) + Math.abs(point[1]-dragStart[1])
  if (distance > .006) dragging = true
  if (!dragging) return
  previewBox = [Math.min(dragStart[0], point[0]), Math.min(dragStart[1], point[1]),
                Math.max(dragStart[0], point[0]), Math.max(dragStart[1], point[1])]
  drawCanvas()
  const context = $('labelCanvas').getContext('2d')
  context.strokeStyle = '#fff'
  context.setLineDash([4,3])
  context.strokeRect(previewBox[0]*$('labelCanvas').width, previewBox[1]*$('labelCanvas').height,
    (previewBox[2]-previewBox[0])*$('labelCanvas').width, (previewBox[3]-previewBox[1])*$('labelCanvas').height)
  context.setLineDash([])
})

$('labelCanvas').addEventListener('pointerup', event => {
  if (!dragStart) return
  const point = canvasPoint(event)
  const box = [Math.min(dragStart[0], point[0]), Math.min(dragStart[1], point[1]),
               Math.max(dragStart[0], point[0]), Math.max(dragStart[1], point[1])]
  const moved = dragging
  dragStart = null; previewBox = null; dragging = false
  if (!moved){
    if (pressBox >= 0){
      state.selected = pressBox
      drawCanvas(); drawBoxList()
    }
    pressBox = -1
    return
  }
  if ((box[2]-box[0]) > 0.004 && (box[3]-box[1]) > 0.004){
    const key = $('boxClass').value
    if (key){
      state.boxes.push({class:key, x1:box[0], y1:box[1], x2:box[2], y2:box[3]})
      state.selected = state.boxes.length - 1
    }
  }
  pressBox = -1
  drawCanvas(); drawBoxList()
})

$('labelCanvas').addEventListener('pointerleave', () => { state.hoverBox = -1; hideHoverInfo() })

$('labelCanvas').addEventListener('contextmenu', event => {
  event.preventDefault()
  if (!state.img) return
  const boxIndex = boxAt(canvasPoint(event))
  if (boxIndex >= 0){
    state.boxes.splice(boxIndex, 1)
    state.selected = -1
    state.hoverBox = -1
    hideHoverInfo()
    drawCanvas(); drawBoxList()
  }
})

function drawBoxList(){
  const list = $('boxList')
  list.replaceChildren()
  state.boxes.forEach((box, index) => {
    const row = document.createElement('div')
    row.className = 'box-row'
    row.innerHTML = '<span class="swatch" style="background:' + colorOf(box.class) + '"></span>'
      + '<select></select><button class="quiet">删除</button>'
    const select = row.querySelector('select')
    fillSelect(select, state.meta.det_classes, {empty:null})
    select.value = box.class
    select.onchange = () => { box.class = select.value; drawCanvas(); drawBoxList() }
    row.querySelector('button').onclick = () => { state.boxes.splice(index,1); state.selected=-1; drawCanvas(); drawBoxList() }
    row.onclick = event => { if (event.target === row || event.target.tagName === 'SPAN'){ state.selected = index; drawCanvas() } }
    list.append(row)
  })
  if (!state.boxes.length) list.innerHTML = '<p class="muted">还没有框。拖动鼠标画框。</p>'
}

$('deleteBox').onclick = () => {
  if (state.selected < 0) return
  state.boxes.splice(state.selected, 1)
  state.selected = -1
  drawCanvas(); drawBoxList()
}

$('saveLabels').onclick = async () => {
  if (!state.image) return toast('先选择图片', true)
  const annotator = $('annotator').value.trim()
  localStorage.setItem('pcbPlatformAnnotator', annotator)
  const savedId = state.image.id
  try{
    state.image = await api('/api/images/' + savedId + '/labels', {
      boxes: state.boxes.map(box => ({class:box.class, x1:box.x1, y1:box.y1, x2:box.x2, y2:box.y2})),
      annotator,
    })
    const index = state.images.findIndex(image => image.id === savedId)
    await loadLabelImages()
    if (index >= 0 && index < state.images.length - 1){
      await openImage(state.images[index + 1].id)
      toast('已保存，自动打开下一张')
    } else {
      toast('已保存。这是最后一张，已全部存好。')
    }
  } catch(error){ toast(error.message, true) }
}

window.addEventListener('resize', () => { if (currentView() === 'label') drawCanvas() })

/* --------------------------------------------------------------- datasets */

async function loadDatasets(){
  state.datasets = await api('/api/projects/' + state.project.id + '/datasets')
  const list = $('datasetList')
  list.replaceChildren()
  for (const dataset of state.datasets){
    const counts = safeJson(dataset.counts, {})
    const warnings = safeJson(dataset.warnings, [])
    const card = document.createElement('article')
    card.className = 'panel'
    card.innerHTML = '<div class="panelhead"><h2>v' + dataset.version + ' · ' + esc(state.meta.templates[dataset.template]?.label || dataset.template) + '</h2>'
      + '<small>' + date(dataset.created) + '</small></div>'
      + '<div class="metric"><div><span>训练</span><strong>' + (counts.train || 0) + '</strong></div>'
      + '<div><span>验证</span><strong>' + (counts.val || 0) + '</strong></div>'
      + '<div><span>测试</span><strong>' + (counts.test || 0) + '</strong></div>'
      + '<div><span>标注框</span><strong>' + (counts.boxes || 0) + '</strong></div></div>'
      + Object.entries(counts.splits || {}).map(([split,stats])=>'<p>'+esc(split)+' · 独立板 '+stats.boards+' · 无框图片 '+stats.normal+' · 缺陷框 '+stats.boxes+'<br>'+Object.entries(stats.classes || {}).map(([key,count])=>esc(labelOf(key))+' '+count).join(' · ')+'</p>').join('')
      + (warnings.length ? '<div class="warnbox">' + warnings.map(esc).join('<br>') + '</div>' : '')
    list.append(card)
  }
  if (!state.datasets.length) list.innerHTML = '<p class="empty">还没有数据集版本。先标注并保存，再创建。</p>'
  $('datasetStats').textContent = state.datasets.length + ' 个版本'
  fillTrainDatasets()
}

function safeJson(value, fallback){
  if (!value) return fallback
  try{ return typeof value === 'string' ? JSON.parse(value) : value } catch { return fallback }
}

$('datasetForm').onsubmit = async event => {
  event.preventDefault()
  const form = event.target
  try{
    const result = await api('/api/projects/' + state.project.id + '/datasets', {
      template: form.elements.template.value,
      split_policy: form.elements.split_policy.value,
      val_ratio: Number(form.elements.val_ratio.value),
      test_ratio: Number(form.elements.test_ratio.value),
      seed: Number(form.elements.seed.value),
      created_by: form.elements.created_by.value,
    })
    toast('数据集 v' + result.version + ' 已创建：train ' + result.counts.train + ' / val ' + result.counts.val + ' / test ' + result.counts.test)
    await loadDatasets()
  } catch(error){ toast(error.message, true) }
}

/* ------------------------------------------------------------------ train */

function fillTrainDatasets(){
  const select = $('trainDataset')
  select.replaceChildren()
  for (const dataset of state.datasets){
    const counts = safeJson(dataset.counts, {})
    select.append(new Option('v' + dataset.version + ' · ' + dataset.template + ' · '
      + (counts.train || 0) + '/' + (counts.val || 0) + '/' + (counts.test || 0), dataset.id))
  }
  if (!state.datasets.length) select.append(new Option('暂无数据集', ''))
}

$('trainDataset').onchange = () => {
  const dataset = state.datasets.find(item => item.id === $('trainDataset').value)
  if (!dataset) return
  const template = state.meta.templates[dataset.template]
  const form = $('trainForm')
  form.elements.imgsz.value = template.imgsz
  form.elements.model.value = template.model
}

$('trainForm').onsubmit = async event => {
  event.preventDefault()
  const form = event.target
  if (!form.elements.dataset_id.value) return toast('请先创建数据集', true)
  try{
    const job = await api('/api/datasets/' + form.elements.dataset_id.value + '/train', {
      epochs: Number(form.elements.epochs.value), imgsz: Number(form.elements.imgsz.value),
      batch: Number(form.elements.batch.value), device: form.elements.device.value,
      model: form.elements.model.value,
    })
    toast('训练任务已启动')
    await loadJobs()
    openJob(job.id)
  } catch(error){ toast(error.message, true) }
}

async function loadJobs(){
  state.jobs = await api('/api/projects/' + state.project.id + '/jobs')
  renderJobs()
  $('trainStats').textContent = state.jobs.length + ' 个任务'
}

function renderJobs(){
  const list = $('jobList')
  list.replaceChildren()
  for (const job of state.jobs){
    const row = document.createElement('div')
    row.className = 'job' + (state.job === job.id ? ' active' : '')
    row.innerHTML = '<div><strong>' + job.template + ' · ' + job.id.slice(0,8) + '</strong>'
      + '<small>' + date(job.created) + ' · ' + job.state + '</small></div>'
      + '<span class="tag ' + job.state + '">' + job.state + '</span>'
    row.onclick = () => openJob(job.id)
    list.append(row)
  }
  if (!state.jobs.length) list.innerHTML = '<p class="empty">还没有训练任务。</p>'
}

async function openJob(id){
  state.job = id
  renderJobs()
  try{
    const job = await api('/api/jobs/' + id)
    $('jobState').textContent = job.state
    $('jobLog').textContent = job.log_tail || '等待日志…'
    $('jobLog').scrollTop = $('jobLog').scrollHeight
    if (job.state === 'running' || job.state === 'queued') {
      clearTimeout(state.timer)
      state.timer = setTimeout(() => openJob(id), 2000)
    } else {
      clearTimeout(state.timer)
      if (job.state === 'done') toast('训练完成，模型已注册')
      if (job.metrics) await loadJobs()
    }
  } catch(error){ toast(error.message, true) }
}

/* ----------------------------------------------------------------- models */

async function loadModels(){
  state.models = await api('/api/projects/' + state.project.id + '/models')
  const list = $('modelList')
  list.replaceChildren()
  for (const model of state.models){
    const metrics = model.metrics || {}
    const summary = model.card?.template === 'det'
      ? 'mAP50 ' + pct(metrics.map50) + ' · mAP ' + pct(metrics.map)
      : 'Top-1 ' + pct(metrics.top1) + ' · Top-5 ' + pct(metrics.top5)
    const card = document.createElement('article')
    card.className = 'panel'
    card.innerHTML = '<div class="panelhead"><h2>' + esc(model.version) + ' · ' + esc(model.card?.template || '') + '</h2>'
      + (model.deployed_station ? '<span class="tag done">已部署 ' + esc(model.deployed_station) + '</span>' : '') + '</div>'
      + '<p>' + summary + ' · ' + date(model.created) + '</p>'
      + '<small>SHA256 ' + esc(model.sha256.slice(0,16)) + '…</small>'
      + '<div class="metric" id="metric-' + model.id + '"></div>'
      + '<div class="toolbar"><a class="badge export" href="' + base + '/api/models/' + model.id + '/export">导出训练模型</a>'
      + '<a class="badge" href="' + base + '/api/models/' + model.id + '/file">仅下载权重</a>'
      + '<button class="primary publish">发布到工位</button></div>'
    list.append(card)
    renderMetrics(card.querySelector('#metric-' + model.id), model)
    card.querySelector('.publish').onclick = async () => {
      const station = prompt('发布到哪个工位？', state.project.station || '')
      if (station === null) return
      try{ await api('/api/models/' + model.id + '/publish', {station}); toast('已发布'); await loadModels() }
      catch(error){ toast(error.message, true) }
    }
  }
  if (!state.models.length) list.innerHTML = '<p class="empty">还没有模型。训练成功后会自动注册。</p>'
  $('modelStats').textContent = state.models.length + ' 个模型版本'
}

$('autoTrain').onclick = async () => {
  if (!needProject()) return
  const button = $('autoTrain')
  button.disabled = true
  $('autoTrainState').textContent = '准备训练…'
  $('autoTrainLog').textContent = ''
  try{
    const dataset = await api('/api/projects/' + state.project.id + '/datasets', {
      template: 'det', split_policy: 'board', val_ratio: .2, test_ratio: .2,
      seed: 20261110, created_by: 'auto',
    })
    const job = await api('/api/datasets/' + dataset.id + '/train', {
      epochs: Number($('autoEpochs').value), imgsz: Number($('autoImgsz').value),
      batch: Number($('autoBatch').value), device: 'cpu',
      auto_publish: true, station: state.project.station || '',
    })
    $('autoTrainState').textContent = '训练中 ' + job.id.slice(0, 8)
    let detail = job
    while (!['done', 'failed', 'cancelled'].includes(detail.state)){
      await new Promise(resolve => setTimeout(resolve, 3000))
      detail = await api('/api/jobs/' + job.id)
      $('autoTrainState').textContent = detail.state
      $('autoTrainLog').textContent = detail.log_tail || ''
    }
    if (detail.state !== 'done') throw Error(detail.error || '训练失败')
    await loadModels()
    const trained = state.models.find(model => model.job_id === job.id)
    if (trained?.deployed_station){
      $('autoTrainState').textContent = '已自动启用'
      $('autoTrainLog').textContent = '模型 ' + trained.version + ' 已自动发布到工位，检测端下次识别时加载。'
      toast('模型已自动启用，检测端下次识别时生效')
    } else {
      $('autoTrainState').textContent = '训练完成，未启用'
      $('autoTrainLog').textContent = '模型已生成，但验证或独立测试数据不足，或 mAP50=0，未自动发布。请增加已标注图片后重试。'
      toast('训练完成但未自动启用：验证或测试数据不足，或指标为 0', true)
    }
    await loadAccuracy()
  } catch(error){
    $('autoTrainState').textContent = '失败'
    $('autoTrainLog').textContent = error.message
    toast(error.message, true)
  } finally {
    button.disabled = false
  }
}

function pct(value){
  return value === null || value === undefined ? '—' : (100*value).toFixed(1) + '%'
}

function renderMetrics(container, model){
  if (!container) return
  const metrics = model.metrics || {}
  const per = metrics.per_class || {}
  const rows = Object.entries(per).map(([key, value]) =>
    '<div><span>' + esc(labelOf(key)) + '</span><strong>' + (value.recall != null ? pct(value.recall) : pct(value.ap50)) + '</strong></div>').join('')
  const test = metrics.test
  const comparison = test ? '<p>独立测试 · '+(test.evaluation_status === 'unavailable' ? esc(test.reason) : 'mAP50 '+pct(test.map50)+' · '+test.val_images+' 张图 / '+test.instances+' 个缺陷')+'</p>'
    + Object.entries(test.per_class || {}).map(([key,item])=>'<p>'+esc(labelOf(key))+' · '+(item.support ? '样本 '+item.support+' · 漏检 '+item.missed+' · 背景误报 '+item.false_positive : '未评估')+'</p>').join('') : '<p>此版本未记录独立测试结果</p>'
  const normal = test?.normal_images != null ? '<p>正常板误报 '+test.normal_false_positive_images+' / '+test.normal_images+' 张（置信度阈值 25%）</p>' : ''
  const baseline = metrics.baseline_test
  container.innerHTML = (rows || '<div><span>指标</span><strong>—</strong></div>') + comparison + normal
    + (baseline ? '<p>同一测试集线上版本 '+esc(metrics.baseline_version)+' · mAP50 '+pct(baseline.map50)+' · 正常板误报 '+(baseline.normal_false_positive_images ?? '—')+' / '+(baseline.normal_images ?? '—')+'</p>'+Object.entries(baseline.per_class || {}).map(([key,item])=>'<p>线上 '+esc(labelOf(key))+' · 漏检 '+(item.missed ?? '—')+' · 背景误报 '+(item.false_positive ?? '—')+'</p>').join('') : '')
    + (metrics.comparison_reason ? '<p>'+esc(metrics.comparison_reason)+'</p>' : '')
}

/* ---------------------------------------------------------------- monitor */

async function loadMonitor(){
  const data = await api('/api/projects/' + state.project.id + '/monitor')
  const card = $('deploymentCard')
  const deployment = data.deployment
  card.innerHTML = (data.deployments || []).map(item=>'<div><h3>'+esc(item.station)+' · 指定版本 '+esc(item.version)+'</h3><p>现场加载 '+(item.loaded_model === item.model_id && !item.load_error ? '已加载指定版本' : item.load_error ? esc(item.load_error) : '等待检测端加载')+' · '+esc(item.loaded_at || '尚无回执')+'</p></div>').join('') || '<p>还没有发布模型。</p>'
  $('monitorStats').textContent = '反馈 ' + data.feedback_count + ' · 待处理 ' + data.open
  renderFeedback(data.feedback)
}

function renderFeedback(items){
  const list = $('feedbackList')
  list.replaceChildren()
  for (const item of items){
    const row = document.createElement('div')
    row.className = 'box-row'
    row.innerHTML = '<span>' + esc(item.predicted || '未填预测') + '</span><small>' + esc(item.note || '') + '</small>'
      + '<span class="tag ' + (item.state === 'converted' ? 'done' : 'queued') + '">' + (item.state === 'converted' ? '已转标注' : '待处理') + '</span>'
      + '<button class="quiet" ' + (item.state === 'converted' ? 'disabled' : '') + '>转标注</button>'
    row.querySelector('button').onclick = async () => {
      try{ await api('/api/feedback/' + item.id + '/convert', {}); toast('已转为待标注图片'); await loadMonitor() }
      catch(error){ toast(error.message, true) }
    }
    list.append(row)
  }
  if (!items.length) list.innerHTML = '<p class="muted">暂无反馈。</p>'
}

$('feedbackForm').onsubmit = async event => {
  event.preventDefault()
  const form = event.target
  const data = new FormData(form)
  try{
    await api('/api/projects/' + state.project.id + '/feedback', data)
    toast('已加入反馈队列')
    form.reset()
    await loadMonitor()
  } catch(error){ toast(error.message, true) }
}

/* ------------------------------------------------------------------- boot */

$('tabs').addEventListener('click', event => {
  const button = event.target.closest('button[data-view]')
  if (!button) return
  location.hash = button.dataset.view
})
window.addEventListener('hashchange', () => switchView(currentView()))
$('helpBtn').onclick = showHelp
$('homeHelp').onclick = showHelp
$('closeHelp').onclick = () => $('helpDialog').close()
document.querySelectorAll('[data-go]').forEach(button => {
  button.onclick = () => { location.hash = button.dataset.go }
})
$('projectSelect').onchange = event => selectProject(event.target.value)
$('newProjectBtn').onclick = () => { location.hash = 'projects'; $('projectForm').elements.name.focus() }
$('projectForm').onsubmit = async event => {
  event.preventDefault()
  const form = event.target
  try{
    const result = await api('/api/projects', {
      name: form.elements.name.value, board_model: form.elements.board_model.value,
      station: form.elements.station.value, note: form.elements.note.value,
    })
    form.reset()
    await loadProjects()
    await selectProject(result.id)
    toast('项目已创建')
  } catch(error){ toast(error.message, true) }
}
$('uploadForm').onsubmit = async event => {
  event.preventDefault()
  const form = event.target
  const files = form.elements.files.files
  if (!files || !files.length) return toast('请选择图片', true)
  const data = new FormData(form)
  try{
    const result = await api('/api/projects/' + state.project.id + '/images', data)
    localStorage.setItem('pcbPlatformBatch', form.elements.batch.value)
    localStorage.setItem('pcbPlatformBoard', form.elements.board_serial.value)
    form.elements.files.value = ''
    toast('已入库 ' + result.saved.length + ' 张，正在打开框缺陷…')
    await loadProject(); await loadImages()
    setTimeout(() => { if (currentView() === 'capture') location.hash = 'label' }, 500)
  } catch(error){ toast(error.message, true) }
}
document.querySelectorAll('[data-img-filter]').forEach(button => button.onclick = () => {
  state.filter = button.dataset.imgFilter
  loadImages().catch(error => toast(error.message, true))
})
$('datasetTemplate').onchange = () => {
  const template = state.meta.templates[$('datasetTemplate').value]
  const form = $('datasetForm')
  if (template) form.elements.split_policy.focus()
}
$('projectSelect').value = ''

;(async () => {
  try{
    state.meta = await api('/api/meta')
    fillStaticSelects()
    await loadProjects()
    const saved = localStorage.getItem('pcbPlatformProject')
    const selected = state.projects.find(project => project.id === saved) || state.projects[0]
    if (selected) await selectProject(selected.id)
    const view = currentView()
    if (!['home','projects','capture','label','review','accuracy','datasets','train','models','monitor'].includes(view)) location.hash = 'home'
    else switchView(view)
    if (!localStorage.getItem('pcbPlatformHelpSeen')) setTimeout(showHelp, 800)
  } catch(error){ toast(error.message, true) }
})()

const base = location.pathname.startsWith('/platform') ? '/platform' : ''
const $ = id => document.getElementById(id)
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))
const COLORS = ['#38bdf8','#f59e0b','#a78bfa','#f87171','#4ade80','#22d3ee','#fb7185','#facc15']
const state = {
  meta:null, project:null, images:[], image:null, boxes:[], selected:-1,
  filter:'all', img:null, modelLoading:false, imageLimit:200, listLoading:false, listRequest:0,
  hoverBox:-1,
  viewport:{x:0,y:0,w:1,h:1}, reference:null, alignment:null, overview:null,
  annotationDirty:false, imageRequest:0, alignmentRequest:0,
  siteTemplate:null,siteDirty:false,editingSites:false,selectedSite:null,
  selectedSites:[],
  sitePopup:null,
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

async function refreshWorkspace(){
  await loadProject()
  await loadLabelImages()
  await loadModelUpdate()
}

async function loadProject(){
  if (state.project) state.project = await api('/api/projects/' + state.project.id)
}

const isProcessed = image => image && image.state !== 'new' && image.review_status === 'approved'

function updateEditorAccess(){
  const locked = !state.img || isProcessed(state.image)
  document.querySelector('.detail-tools').inert = !state.img
  document.querySelector('.board-overview').inert = !state.img
  document.querySelector('.site-tools').inert = !state.img
  $('siteHint').hidden = !!locked
  for (const id of ['annotator','saveLabels']) $(id).disabled = !!locked
  $('deleteTrainingImage').disabled = !state.img
  $('setReference').hidden = locked || !!state.reference
  $('manualAlign').hidden = !state.img || !state.reference || state.reference.id === state.image?.id || !!state.alignment
  $('readOnlyHint').hidden = !isProcessed(state.image)
}

function resetEditor(){
  state.image=null;state.img=null;state.boxes=[];state.selected=-1
  state.annotationDirty=false;state.siteDirty=false;state.editingSites=false
  state.selectedSite=null;state.selectedSites=[];state.overview=null
  ++state.imageRequest;++state.alignmentRequest
  closeSiteDefectPopup()
  for (const id of ['labelCanvas','overviewCanvas']){
    const canvas=$(id)
    canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height)
  }
  $('editorEmpty').hidden=false
  document.querySelector('.annotation-workspace').hidden=true
  $('deleteTrainingImage').disabled=true
  $('saveLabels').disabled=true
  updateEditorAccess()
  renderSites()
  updateLabelProgress()
}

async function deleteTrainingImage(id) {
  if (state.savingAnnotation) return
  try {
    const result = await api('/api/images/' + id, undefined, 'DELETE')
    if (state.image?.id === id) resetEditor()
    await loadProject()
    await loadLabelImages(false)
    if (!state.image && state.images.length) await openImage(state.images[0].id)
    await loadModelUpdate()
    toast(result.retained_in_versions ? '已从图库删除，历史数据集保留该图片快照' : '图片已删除')
  } catch(error) { toast(error.message, true) }
}
$('deleteTrainingImage').onclick = () => {
  if (state.image) deleteTrainingImage(state.image.id)
  else toast('请先选择图片',true)
}

/* --------------------------------------------------------------- labeling */

async function loadLabelImages(openCurrent=true){
  const requestId=++state.listRequest
  state.listLoading=true
  const query=new URLSearchParams({limit:'200'})
  if (state.filter==='new') query.set('processing','unprocessed')
  else if (state.filter==='approved') query.set('processing','processed')
  const images=[]
  try{
    while (images.length<state.imageLimit){
      query.set('offset',String(images.length))
      const page=await api('/api/projects/'+state.project.id+'/images?'+query)
      images.push(...page)
      if (page.length<200) break
    }
    if (requestId!==state.listRequest) return
    state.images=images
  }finally{if(requestId===state.listRequest) state.listLoading=false}
  $('loadMoreImages').hidden=images.length<state.imageLimit
  document.querySelectorAll('[data-img-filter]').forEach(button=>button.classList.toggle('active',button.dataset.imgFilter===state.filter))
  const list = $('labelList')
  list.replaceChildren()
  for (const image of state.images){
    const processed=isProcessed(image)
    const row = document.createElement('div')
    row.className = 'row '+(processed ? 'image-processed' : 'image-unprocessed')+(state.image?.id === image.id ? ' active' : '')
    row.dataset.id = image.id
    row.innerHTML = '<img loading="lazy" src="' + base + '/api/images/' + image.id + '/file?v=' + image.revision + '&w=160" alt="">'
      + '<div><strong>' + esc(image.filename) + '</strong><small>' + (processed ? '已处理' : '未处理') + ' · '
      + (image.label_count || 0) + ' 框</small></div>'
    row.tabIndex=0
    row.setAttribute('role','button')
    row.setAttribute('aria-label',image.filename+' · '+(processed ? '已处理' : '未处理'))
    row.onkeydown=event=>{if(event.key==='Enter' || event.key===' '){event.preventDefault();row.click()}}
    row.onclick = () => openImage(image.id).catch(error => toast(error.message,true))
    list.append(row)
  }
  if (!state.images.length) list.innerHTML='<p class="empty">暂无图片<br>点击“导入图片”添加训练样本</p>'
  $('labelStats').textContent = state.images.length + ' 张图片'
  if (!openCurrent) return updateLabelProgress()
  if (state.image && state.images.some(image => image.id === state.image.id)) await openImage(state.image.id)
  else if (state.images.length) await openImage(state.images[0].id)
  else resetEditor()
  updateLabelProgress()
}

async function refreshImageLibrary(){
  if (document.hidden || !state.project || state.listLoading || state.annotationDirty || state.siteDirty || state.savingAnnotation || state.importingImages) return
  await loadLabelImages(false)
  if (!state.image && state.images.length) await openImage(state.images[0].id)
}

$('loadMoreImages').onclick=async()=>{
  state.imageLimit+=200
  try{await loadLabelImages(false)}catch(error){toast(error.message,true)}
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
  if (next >= 0 && next < state.images.length) openImage(state.images[next].id).catch(error=>toast(error.message,true))
}

$('prevImage').onclick = () => moveImage(-1)
$('nextImage').onclick = () => moveImage(1)

async function openImage(id){
  if (!leaveAnnotation()) return
  closeSiteDefectPopup()
  const requestId = ++state.imageRequest
  ++state.alignmentRequest
  state.img = null
  state.image = null
  updateEditorAccess()
  state.overview = null
  state.alignment = null
  $('labelCanvas').getContext('2d').clearRect(0,0,$('labelCanvas').width,$('labelCanvas').height)
  $('overviewCanvas').getContext('2d').clearRect(0,0,$('overviewCanvas').width,$('overviewCanvas').height)
  let detail = await api('/api/images/' + id)
  if (requestId !== state.imageRequest || detail.project_id !== state.project?.id) return
  state.image = detail
  state.boxes = []
  state.viewport = {x:0,y:0,w:1,h:1}
  state.annotationDirty = false
  dragStart = null
  previewBox = null
  dragging = false
  $('detailBrightness').value = 100
  const view = detail.annotation_view || {}
  state.alignmentDisabled = false
  state.selected = -1
  state.hoverBox = -1
  hideHoverInfo()
  $('annotator').value = state.image.labels[0]?.annotator || localStorage.getItem('pcbPlatformAnnotator') || ''
  const [reference, template] = await Promise.all([
    api('/api/projects/' + detail.project_id + '/alignment-reference'),
    api('/api/projects/' + detail.project_id + '/solder-template'),
  ])
  if (requestId !== state.imageRequest) return
  let preview = null
  const hint = $('alignmentHint')
  hint.hidden = true
  if (reference && reference.id !== id && !detail.alignment){
    hint.hidden = false
    hint.textContent = '正在匹配板面位置…'
    try{
      const saved = view.alignment
      preview = await api('/api/images/'+id+'/alignment',{
        reference_id:reference.id, include_preview:false, matrix:saved?.reference_id === reference.id ? saved.matrix : undefined,
      })
      if (requestId !== state.imageRequest) return
      hint.hidden = true
    }catch(error){
      if (requestId !== state.imageRequest) return
      hint.textContent = error.message
    }
    detail = await api('/api/images/' + id)
    if (requestId !== state.imageRequest) return
  }
  const url = base + '/api/images/' + id + '/file?v=' + detail.revision
  const displayUrl = detail.alignment ? url : preview?.preview || url
  const image = await loadAnnotationImage(displayUrl)
  if (requestId !== state.imageRequest) return
  state.image = detail
  state.boxes = detail.labels.map(item => ({...item}))
  state.alignment = detail.alignment || (preview?.preview ? {reference_id:preview.reference_id,matrix:preview.matrix} : null)
  state.img = image
  const thumbnail = document.querySelector('#labelList .row[data-id="'+id+'"] img')
  if (thumbnail) thumbnail.src = displayUrl === url ? url + '&w=160' : displayUrl
  $('editorEmpty').hidden=true
  document.querySelector('.annotation-workspace').hidden=false
  state.reference = reference
  updateEditorAccess()
  state.siteTemplate=template
  state.siteDirty=false
  state.editingSites=false
  state.selectedSite=null
  state.selectedSites=[]
  $('siteSearch').value=''
  state.overview = image
  document.querySelectorAll('#labelList .row').forEach(row => row.classList.toggle('active', row.dataset.id === id))
  updateLabelProgress()
  drawCanvas()
}

$('setReference').onclick = async () => {
  if (!state.img || isProcessed(state.image) || !leaveAnnotation()) return
  const id = state.image.id
  try{
    await api('/api/projects/'+state.project.id+'/alignment-reference',{image_id:id,only_if_empty:true})
    if (state.image?.id === id) await openImage(id)
  }catch(error){toast(error.message,true)}
}

let alignmentDraft = null
function drawAlignmentPoints(){
  const draft = alignmentDraft
  const count = draft.points.length
  const source = count < 4
  const picture = source ? draft.source : draft.reference
  const canvas = $('pointCanvas')
  const scale = Math.min(1, 900/picture.naturalWidth, 540/picture.naturalHeight)
  canvas.width = Math.max(1,Math.round(picture.naturalWidth*scale))
  canvas.height = Math.max(1,Math.round(picture.naturalHeight*scale))
  const ctx = canvas.getContext('2d')
  ctx.drawImage(picture,0,0,canvas.width,canvas.height)
  draft.points.slice(source ? 0 : 4).forEach(([x,y],i)=>{
    ctx.fillStyle='#12324a';ctx.strokeStyle='#fff'
    ctx.beginPath();ctx.arc(x*canvas.width,y*canvas.height,9,0,Math.PI*2);ctx.fill();ctx.stroke()
    ctx.fillStyle='#fff';ctx.textAlign='center';ctx.fillText(i+1,x*canvas.width,y*canvas.height+3)
  })
  $('pointInstruction').textContent = count === 8 ? '已选好四组对应点，可完成对齐。'
    : (source ? '当前图片：顺时针选择板面四角' : '标准图：按相同顺序选择对应四角')+'（第 '+(count%4+1)+' 点）'
  $('undoPoint').disabled = count === 0
  $('applyAlignment').disabled = count !== 8
}

$('manualAlign').onclick = async () => {
  if (!state.img || !state.reference || state.alignment) return
  const request = state.imageRequest, referenceId = state.reference.id
  try{
    const reference = await loadAnnotationImage(base+'/api/projects/'+state.project.id+'/alignment-reference/file?id='+encodeURIComponent(referenceId))
    if (request !== state.imageRequest) return
    alignmentDraft = {source:state.img, reference, referenceId, id:state.image.id, points:[]}
    drawAlignmentPoints()
    $('alignmentDialog').showModal()
  }catch(error){toast(error.message,true)}
}
$('pointCanvas').onclick = event => {
  if (!alignmentDraft || alignmentDraft.busy || alignmentDraft.points.length >= 8) return
  const rect = event.currentTarget.getBoundingClientRect()
  alignmentDraft.points.push([(event.clientX-rect.left)/rect.width,(event.clientY-rect.top)/rect.height])
  drawAlignmentPoints()
}
$('undoPoint').onclick = () => {if (!alignmentDraft.busy){alignmentDraft.points.pop(); drawAlignmentPoints()}}
$('cancelAlignment').onclick = () => $('alignmentDialog').close()
$('alignmentDialog').oncancel = event => {if (alignmentDraft?.busy) event.preventDefault()}
$('alignmentDialog').onclose = () => {alignmentDraft=null}
$('applyAlignment').onclick = async () => {
  const draft = alignmentDraft
  if (!draft || draft.busy || draft.points.length !== 8) return
  if ((state.annotationDirty || state.siteDirty) && !confirm("完成对齐将重新打开图片并丢弃未保存的修改，继续吗？")) return
  draft.busy=true
  $('applyAlignment').disabled=true
  $('cancelAlignment').disabled=true
  try{
    await api('/api/images/'+draft.id+'/alignment',{reference_id:draft.referenceId,include_preview:false,
      source_points:draft.points.slice(0,4),reference_points:draft.points.slice(4)})
    state.annotationDirty=false
    state.siteDirty=false
    $('alignmentDialog').close()
    await openImage(draft.id)
  }catch(error){$('pointInstruction').textContent=error.message}
  finally{draft.busy=false;$('cancelAlignment').disabled=false;$('applyAlignment').disabled=false}
}

function boxAt(point){
  for (let index = state.boxes.length - 1; index >= 0; index--){
    const polygon = displayPolygon(state.boxes[index])
    const turns = polygon.map((a,i)=>{
      const b = polygon[(i+1)%polygon.length]
      return (b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0])
    })
    if (turns.length && (turns.every(turn=>turn>=-1e-10) || turns.every(turn=>turn<=1e-10))) return index
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
  const bounds = boundsOf(displayPolygon(box))
  if (!bounds) { info.hidden=true; return }
  const canvas = $('labelCanvas'), wrap = canvas.parentElement
  const canvasRect = canvas.getBoundingClientRect(), wrapRect = wrap.getBoundingClientRect()
  info.textContent = '缺陷：' + labelOf(box.class)
  info.hidden = false
  const width = info.offsetWidth
  const v = state.viewport
  const left = canvasRect.left - wrapRect.left + (bounds.x1-v.x)/v.w * canvasRect.width
  const top = canvasRect.top - wrapRect.top + (bounds.y1-v.y)/v.h * canvasRect.height - 30
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
  const image = state.overview || state.img
  const wrap = canvas.parentElement
  const maxWidth = Math.max(280, wrap.clientWidth - 24)
  const maxHeight = Math.max(200, wrap.clientHeight - 20)
  const ratio = Math.min(maxWidth / image.naturalWidth, maxHeight / image.naturalHeight, 2)
  const imageW = Math.max(1, Math.round(image.naturalWidth * ratio))
  const imageH = Math.max(1, Math.round(image.naturalHeight * ratio))
  canvas.width = imageW
  canvas.height = imageH
  const context = canvas.getContext('2d')
  context.clearRect(0, 0, imageW, imageH)
  const v = state.viewport
  context.filter = `brightness(${$('detailBrightness').value}%)`
  context.drawImage(image, v.x*image.naturalWidth, v.y*image.naturalHeight,
    v.w*image.naturalWidth, v.h*image.naturalHeight, 0, 0, imageW, imageH)
  context.filter = 'none'
  drawSiteOverlay(context,imageW,imageH)
  state.boxes.forEach((box, index) => {
    const color = colorOf(box.class)
    context.lineWidth = index === state.selected ? 3.5 : 2.4
    context.strokeStyle = color
    context.fillStyle = color + '22'
    const points = displayPolygon(box)
    if (!points.length) return
    context.beginPath()
    points.forEach(([x,y],i)=>{
      const cx=(x-v.x)/v.w*imageW, cy=(y-v.y)/v.h*imageH
      if (i===0) context.moveTo(cx,cy)
      else context.lineTo(cx,cy)
    })
    context.closePath(); context.fill(); context.stroke()
  })
  $('zoomValue').textContent = (1/v.w).toFixed(1).replace('.0','') + '×'
  drawOverview()
  renderSites()
}

function canvasPoint(event){
  const canvas = $('labelCanvas')
  const rect = canvas.getBoundingClientRect()
  const v = state.viewport
  return [v.x + Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width))*v.w,
          v.y + Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height))*v.h]
}

let dragStart = null
let previewBox = null
let pressBox = -1
let dragging = false

$('labelCanvas').addEventListener('pointerdown', event => {
  if (!state.img || event.button!==0) return
  const point = canvasPoint(event)
  if (isProcessed(state.image)){
    const site=siteAt(point)
    if (site) inspectSite(site)
    return
  }
  pressBox = boxAt(point)
  state.selected = -1
  dragging = false
  dragStart = point
  $('labelCanvas').setPointerCapture(event.pointerId)
})

$('labelCanvas').addEventListener('pointermove', event => {
  if (!state.img) return
  if (!dragStart){
    const boxIndex = boxAt(canvasPoint(event))
    if (boxIndex !== state.hoverBox){
      state.hoverBox = boxIndex
      showHoverInfo()
    }
    $('labelCanvas').style.cursor = boxIndex >= 0 || siteAt(canvasPoint(event)) ? 'pointer' : isProcessed(state.image) ? 'default' : 'crosshair'
    return
  }
  const point = canvasPoint(event)
  const distance = Math.abs(point[0]-dragStart[0]) + Math.abs(point[1]-dragStart[1])
  if (distance/state.viewport.w > .006) dragging = true
  if (!dragging) return
  previewBox = [Math.min(dragStart[0], point[0]), Math.min(dragStart[1], point[1]),
                Math.max(dragStart[0], point[0]), Math.max(dragStart[1], point[1])]
  drawCanvas()
  const context = $('labelCanvas').getContext('2d')
  context.strokeStyle = '#fff'
  context.setLineDash([4,3])
  const v = state.viewport
  context.strokeRect((previewBox[0]-v.x)/v.w*$('labelCanvas').width, (previewBox[1]-v.y)/v.h*$('labelCanvas').height,
    (previewBox[2]-previewBox[0])/v.w*$('labelCanvas').width, (previewBox[3]-previewBox[1])/v.h*$('labelCanvas').height)
  context.setLineDash([])
})

$('labelCanvas').addEventListener('pointerup', event => {
  if (event.button!==0) return
  if (!dragStart) return
  const point = canvasPoint(event)
  const box = [Math.min(dragStart[0], point[0]), Math.min(dragStart[1], point[1]),
               Math.max(dragStart[0], point[0]), Math.max(dragStart[1], point[1])]
  const moved = dragging
  dragStart = null; previewBox = null; dragging = false
  if (!moved){
    if (state.editingSites){
      editSiteAt(point)
      return
    }
    const site=siteAt(point)
    if (pressBox >= 0){
      openBoxDefectPopup(pressBox)
    } else if (site){
      selectAnnotationSite(site)
    }
    pressBox = -1
    return
  }
  if ((box[2]-box[0])/state.viewport.w > 0.004 && (box[3]-box[1])/state.viewport.h > 0.004){
    if (state.editingSites){
      if (boxFromDisplay(box,'')) addTemplateSite(box)
      else toast('该位置超出原图拍摄范围',true)
      drawCanvas()
      return
    }
    const sites=sitesInRectangle(box)
    const ids=sites.map(site=>site.id)
    state.selectedSites=event.ctrlKey || event.metaKey || event.shiftKey ? [...new Set([...state.selectedSites,...ids])] : ids
    state.selectedSite=state.selectedSites.at(-1) || null
  }
  pressBox = -1
  drawCanvas()
})

$('labelCanvas').addEventListener('pointerleave', () => { state.hoverBox = -1; hideHoverInfo() })

$('saveLabels').onclick = async () => {
  if (!state.image || !state.img || state.savingAnnotation) return toast('请等待图片加载完成', true)
  const annotator = $('annotator').value.trim()
  localStorage.setItem('pcbPlatformAnnotator', annotator)
  const savedId = state.image.id
  const revision = state.image.revision
  $('saveLabels').disabled = true
  $('prevImage').disabled = true
  $('nextImage').disabled = true
  state.savingAnnotation = true
  $('annotationWorkspace').inert = true
  $('importTrainingImages').disabled = true
  try{
    if (state.siteDirty) await persistSites()
    const saved = await api('/api/images/' + savedId + '/labels', {
      boxes: state.boxes.map(box => ({class:box.class, x1:box.x1, y1:box.y1, x2:box.x2, y2:box.y2,polygon:box.polygon || []})),
      annotator,
      revision,
      primary_class:state.image.primary_class,
      note:state.image.note,
      annotation_view:{
        alignment:state.alignment ? {matrix:state.alignment.matrix,reference_id:state.alignment.reference_id} : null,
        alignment_disabled:!!state.alignmentDisabled,
        quality_flags:[],
        quality_note:'',
      },
    })
    state.image = saved
    state.selected = -1
    state.selectedSites = []
    state.selectedSite = null
    state.editingSites = false
    closeSiteDefectPopup()
    dragStart=null;dragging=false;previewBox=null
    drawCanvas()
    updateEditorAccess()
    state.annotationDirty = false
    if (state.filter === 'new') state.filter = 'approved'
    await loadProject()
    await loadLabelImages(false)
    await loadModelUpdate()
    toast('已保存')
  } catch(error){ toast(error.message, true) }
  finally { state.savingAnnotation=false; $('importTrainingImages').disabled=false; $('annotationWorkspace').inert=false; updateEditorAccess(); updateLabelProgress() }
}

window.addEventListener('resize', () => {
  drawCanvas();positionSiteDefectPopup()
})

function leaveAnnotation(){
  if (state.savingAnnotation || state.importingImages) return false
  if ((state.annotationDirty || state.siteDirty) && !confirm('当前标注或点位模板尚未保存，确定离开并放弃本次修改？')) return false
  state.annotationDirty = false
  state.siteDirty=false
  return true
}

function loadAnnotationImage(url){
  return new Promise((resolve,reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(Error('图片加载失败，请重新打开'))
    image.src = url
  })
}

function setViewport(x,y,size){
  closeSiteDefectPopup()
  const extent = Math.max(1/24,Math.min(1,size))
  state.viewport = {x:Math.max(0,Math.min(1-extent,x)),y:Math.max(0,Math.min(1-extent,y)),w:extent,h:extent}
  state.hoverBox = -1
  hideHoverInfo()
}

function zoomDetail(factor, point){
  if (!state.img || dragStart) return
  const v = state.viewport
  const anchor = point || [v.x+v.w/2,v.y+v.h/2]
  const extent = Math.max(1/24,Math.min(1,v.w/factor))
  setViewport(anchor[0]-(anchor[0]-v.x)/v.w*extent,anchor[1]-(anchor[1]-v.y)/v.h*extent,extent)
  drawCanvas()
}

$('zoomIn').onclick = () => zoomDetail(1.5)
$('zoomOut').onclick = () => zoomDetail(1/1.5)
$('resetZoom').onclick = () => { setViewport(0,0,1); drawCanvas() }
$('labelCanvas').addEventListener('wheel',event => {
  if (!state.img) return
  event.preventDefault()
  zoomDetail(event.deltaY < 0 ? 1.2 : 1/1.2,canvasPoint(event))
},{passive:false})
$('labelCanvas').addEventListener('pointercancel',() => {
  dragStart=null; previewBox=null; dragging=false; drawCanvas()
})
$('detailBrightness').oninput = drawCanvas
$('resetDisplay').onclick = () => {
  $('detailBrightness').value=100; drawCanvas()
}

function mapPoint(matrix,point){
  const [x,y] = point
  const z = matrix[2][0]*x + matrix[2][1]*y + matrix[2][2]
  if (Math.abs(z) < 1e-9) return null
  return [(matrix[0][0]*x+matrix[0][1]*y+matrix[0][2])/z,
          (matrix[1][0]*x+matrix[1][1]*y+matrix[1][2])/z]
}

function inverseMatrix(m){
  const [a,b,c]=m[0], [d,e,f]=m[1], [g,h,i]=m[2]
  return [[e*i-f*h,c*h-b*i,b*f-c*e],[f*g-d*i,a*i-c*g,c*d-a*f],[d*h-e*g,b*g-a*h,a*e-b*d]]
}

function rectanglePoints(box){
  return [[box.x1,box.y1],[box.x2,box.y1],[box.x2,box.y2],[box.x1,box.y2]]
}

function boundsOf(points){
  if (!points.length) return null
  return {x1:Math.min(...points.map(p=>p[0])),y1:Math.min(...points.map(p=>p[1])),
          x2:Math.max(...points.map(p=>p[0])),y2:Math.max(...points.map(p=>p[1]))}
}

function displayPolygon(box){
  const points = box.polygon?.length===4 ? box.polygon : rectanglePoints(box)
  const projected = points.map(point=>state.alignment ? mapPoint(state.alignment.matrix,point) : point)
  return projected.some(point=>!point || point.some(value=>!Number.isFinite(value))) ? [] : projected
}

function boxFromDisplay(rect,key){
  const points = rectanglePoints({x1:rect[0],y1:rect[1],x2:rect[2],y2:rect[3]})
  const inverse = state.alignment && inverseMatrix(state.alignment.matrix)
  const source = points.map(point=>inverse ? mapPoint(inverse,point) : point)
  if (source.some(point=>!point || point.some(value=>!Number.isFinite(value) || value < -1e-8 || value > 1+1e-8))) return null
  const polygon = source.map(point=>point.map(value=>Math.max(0,Math.min(1,value))))
  return {class:key,...boundsOf(polygon),polygon}
}

function drawOverview(){
  if (!state.overview) return
  const canvas = $('overviewCanvas'), image = state.overview
  const ratio = Math.min(300/image.naturalWidth,380/image.naturalHeight)
  canvas.width = Math.max(1,Math.round(image.naturalWidth*ratio))
  canvas.height = Math.max(1,Math.round(image.naturalHeight*ratio))
  const ctx = canvas.getContext('2d')
  ctx.drawImage(image,0,0,canvas.width,canvas.height)
  const v = state.viewport
  const points = [[v.x,v.y],[v.x+v.w,v.y],[v.x+v.w,v.y+v.h],[v.x,v.y+v.h]]
  if (points.some(point=>!point)) return
  ctx.beginPath()
  points.forEach(([x,y],index) => index ? ctx.lineTo(x*canvas.width,y*canvas.height) : ctx.moveTo(x*canvas.width,y*canvas.height))
  ctx.closePath(); ctx.strokeStyle='#38bdf8'; ctx.lineWidth=2
  ctx.fillStyle='#38bdf822'; ctx.fill(); ctx.stroke()
}

$('overviewCanvas').onclick = event => {
  if (!state.img || !state.overview) return
  const rect = $('overviewCanvas').getBoundingClientRect()
  const point = [(event.clientX-rect.left)/rect.width,(event.clientY-rect.top)/rect.height]
  const extent = Math.min(state.viewport.w,.25)
  setViewport(point[0]-extent/2,point[1]-extent/2,extent)
  drawCanvas()
}

$('annotator').addEventListener('input',()=>{if(state.image) state.annotationDirty=true})

function sitesReady(){
  return !!(state.img && state.reference && state.siteTemplate?.reference_id===state.reference.id
    && (state.image.id===state.reference.id || state.alignment?.reference_id===state.reference.id))
}

function availableSites(){
  if (!sitesReady()) return []
  return state.siteTemplate.sites.filter(site=>boxFromDisplay([site.x1,site.y1,site.x2,site.y2],''))
}

function siteAt(point){
  if (!$('showSites').checked && !state.editingSites) return null
  return availableSites().filter(site=>point[0]>=site.x1 && point[0]<=site.x2 && point[1]>=site.y1 && point[1]<=site.y2)
    .sort((a,b)=>(a.x2-a.x1)*(a.y2-a.y1)-(b.x2-b.x1)*(b.y2-b.y1))[0] || null
}

function drawSiteOverlay(ctx,width,height){
  if (!$('showSites').checked && !state.editingSites) return
  const v=state.viewport
  ctx.save()
  ctx.setLineDash([3,3])
  for (const site of availableSites()){
    if (site.x2<v.x || site.x1>v.x+v.w || site.y2<v.y || site.y1>v.y+v.h) continue
    const selected=site.id===state.selectedSite || state.selectedSites.includes(site.id)
    const x=(site.x1-v.x)/v.w*width,y=(site.y1-v.y)/v.h*height
    const w=(site.x2-site.x1)/v.w*width,h=(site.y2-site.y1)/v.h*height
    ctx.strokeStyle=selected ? '#fbbf24' : '#38bdf8'
    ctx.lineWidth=selected ? 2.5 : 1
    ctx.strokeRect(x,y,w,h)
    if (selected){ctx.fillStyle='#fbbf2422';ctx.fillRect(x,y,w,h)}
    if (selected || v.w<=.25){
      ctx.font='11px sans-serif';ctx.fillStyle=selected ? '#fbbf24' : '#e0f2fe'
      ctx.fillText(site.id,x,Math.max(12,y-3))
    }
  }
  ctx.restore()
}

function renderSites(){
  const available=availableSites()
  const template=state.siteTemplate
  const locked=!state.img || isProcessed(state.image)
  state.selectedSites=state.selectedSites.filter(id=>available.some(site=>site.id===id))
  $('batchSiteDefect').disabled=locked || state.editingSites || !state.selectedSites.length
  $('batchSiteDefect').textContent='选择缺陷类型'+(state.selectedSites.length ? '（'+state.selectedSites.length+'）' : '')
  $('clearSiteSelection').disabled=locked || !state.selectedSites.length
  $('detectSites').disabled=locked || !state.reference
  $('detectSites').textContent=template ? '应用点位模板' : '自动定位焊点 / 焊脚'
  $('editSites').disabled=locked || !sitesReady()
  $('editSites').classList.toggle('active',state.editingSites)
  $('editSites').textContent=state.editingSites ? '结束补点 / 删点' : '补点 / 删点'
  $('saveSites').disabled=locked || !state.siteDirty
  $('siteStatus').textContent=sitesReady()
    ? available.length+' / '+template.sites.length+' 处可定位'+(state.siteDirty ? ' · 模板待保存' : '')
      +(state.selectedSites.length ? ' · 已选 '+state.selectedSites.length+' 处' : state.selectedSite ? ' · 选中 '+state.selectedSite : '')
    : template ? '完成板面对齐后显示点位' : '尚未建立点位模板'
  $('siteHint').textContent=state.editingSites
    ? '左键点击空白位置补点或点击已有点位选中，右键点击点位删除；拖框可设置新点位大小。完成后保存点位模板。'
    : '点击已有缺陷框可修改类型或删除标注；点击或拖框多选点位，再点击“选择缺陷类型”。右键清空选择。'
  const processed=isProcessed(state.image)
  const key=[state.imageRequest,state.alignmentRequest,template?.revision,template?.sites.length,state.siteDirty,processed,
    state.selectedSite,state.selectedSites.join(','),state.editingSites,$('siteSearch').value,available.length].join('|')
  if (state.siteRenderKey===key) return
  state.siteRenderKey=key
  const list=$('siteList'),query=$('siteSearch').value.trim().toLowerCase()
  list.replaceChildren()
  const boxBounds=state.boxes.map(box=>boundsOf(displayPolygon(box)))
  for (const site of available.filter(site=>site.id.toLowerCase().includes(query))){
    const button=document.createElement('button')
    const selected=state.selectedSites.includes(site.id) || site.id===state.selectedSite
    button.className='candidate-row'+(selected ? ' selected' : '')
    button.type='button'
    button.dataset.siteId=site.id
    button.setAttribute('aria-pressed',String(selected))
    button.textContent=(!processed && !state.editingSites ? selected ? '☑ ' : '☐ ' : '')
      +site.id+' · '+siteCondition(siteBoxIndex(site,boxBounds))
    button.onclick=()=>{
      if (processed) inspectSite(site)
      else if (state.editingSites){
        state.selectedSite=site.id;drawCanvas()
      } else selectAnnotationSite(site)
    }
    list.append(button)
  }
}

function addTemplateSite(rect){
  if (!sitesReady()) return
  state.siteTemplate.sites.push({id:'M'+crypto.randomUUID().slice(0,8),x1:rect[0],y1:rect[1],x2:rect[2],y2:rect[3],origin:'manual'})
  state.selectedSite=state.siteTemplate.sites.at(-1).id
  state.siteDirty=true
}

function sitesInRectangle(rect){
  return availableSites().filter(site=>{
    const x=(site.x1+site.x2)/2,y=(site.y1+site.y2)/2
    return x>=rect[0] && x<=rect[2] && y>=rect[1] && y<=rect[3]
  })
}

function removeTemplateSites(ids){
  if (!state.editingSites || !ids.length) return
  state.siteTemplate.sites=state.siteTemplate.sites.filter(site=>!ids.includes(site.id))
  state.selectedSites=state.selectedSites.filter(id=>!ids.includes(id))
  if (ids.includes(state.selectedSite)) state.selectedSite=null
  state.siteDirty=true
  drawCanvas()
}

function editSiteAt(point){
  const site=siteAt(point)
  if (site){state.selectedSite=site.id;drawCanvas();return}
  const sites=state.siteTemplate.sites
  const widths=sites.map(site=>site.x2-site.x1).sort((a,b)=>a-b)
  const heights=sites.map(site=>site.y2-site.y1).sort((a,b)=>a-b)
  const w=widths[Math.floor(widths.length/2)] || .015,h=heights[Math.floor(heights.length/2)] || .025
  const rect=[Math.max(0,point[0]-w/2),Math.max(0,point[1]-h/2),Math.min(1,point[0]+w/2),Math.min(1,point[1]+h/2)]
  if (!boxFromDisplay(rect,'')) return toast('该位置超出原图拍摄范围',true)
  addTemplateSite(rect)
  drawCanvas()
}

function selectAnnotationSite(site){
  const selected=state.selectedSites.includes(site.id)
  state.selectedSites=selected ? state.selectedSites.filter(id=>id!==site.id) : [...state.selectedSites,site.id]
  state.selectedSite=state.selectedSites.at(-1) || null
  closeSiteDefectPopup();drawCanvas()
}

$('batchSiteDefect').onclick=()=>{
  const sites=availableSites().filter(site=>state.selectedSites.includes(site.id))
  if (sites.length) openSiteDefectPopup(sites.at(-1),sites.map(site=>site.id))
}
function clearSiteSelection(){
  state.selectedSites=[];state.selectedSite=null;state.selected=-1
  dragStart=null;previewBox=null;dragging=false
  closeSiteDefectPopup();drawCanvas()
}
$('clearSiteSelection').onclick=clearSiteSelection
document.querySelector('.label-layout').addEventListener('contextmenu',event=>{
  event.preventDefault()
  if (state.savingAnnotation || isProcessed(state.image)) return
  if (state.editingSites){
    const row=event.target.closest('#siteList [data-site-id]')
    const site=event.target===$('labelCanvas') ? siteAt(canvasPoint(event))
      : row ? availableSites().find(item=>item.id===row.dataset.siteId) : null
    if (site){removeTemplateSites([site.id]);return}
  }
  clearSiteSelection()
})

async function persistSites(){
  const template=state.siteTemplate
  const saved=await api('/api/projects/'+state.project.id+'/solder-template',{
    action:'save',reference_id:template.reference_id,revision:template.revision,sites:template.sites,
  })
  state.siteTemplate=saved;state.siteDirty=false
  renderSites()
}

$('detectSites').onclick=async()=>{
  if (!state.img || isProcessed(state.image) || !state.reference || state.savingAnnotation) return
  const pid=state.project.id,reference=state.reference.id
  if (state.image.id!==reference && state.alignment?.reference_id!==reference)
    return toast('请先完成与当前标准图的板面对齐，再应用点位模板',true)
  const endpoint='/api/projects/'+pid+'/solder-template'
  const applying=!!state.siteTemplate
  state.savingAnnotation=true;$('annotationWorkspace').inert=true
  $('siteStatus').textContent=applying ? '正在读取点位模板…' : '正在标准图上定位焊点 / 焊脚…'
  try{
    const template=applying ? await api(endpoint) : await api(endpoint,{
      action:'detect',reference_id:reference,revision:0,
    })
    if (!template || template.reference_id!==reference){
      return toast('标准图或模板已变化，请重新打开图片',true)
    }
    if (applying){
      const previousBoxCount=state.boxes.length
      const sites=[...state.siteTemplate.sites,...template.sites]
      state.boxes=state.boxes.filter(box=>{
        const bounds=boundsOf(displayPolygon(box))
        return !sites.some(site=>siteOverlap(site,bounds)>=.5)
      })
      if (state.boxes.length!==previousBoxCount) state.annotationDirty=true
      state.siteDirty=false
      state.editingSites=false
      state.selectedSites=[]
      state.selectedSite=null
      state.selected=-1
      state.hoverBox=-1
      hideHoverInfo()
      closeSiteDefectPopup()
      $('siteSearch').value=''
    }
    state.siteTemplate=template
    state.siteRenderKey=null
    $('showSites').checked=true
    drawCanvas()
    const count=availableSites().length
    if (!count) return toast(template.sites.length ? '模板点位均在当前图片范围外' : '模板中没有点位',true)
    toast(applying ? '已按模板重置点位，当前显示 '+count+' 处' : '已显示 '+count+' 个模板点位')
  }catch(error){toast(error.message,true)}
  finally{state.savingAnnotation=false;$('annotationWorkspace').inert=false;renderSites()}
}
$('editSites').onclick=()=>{
  if (!sitesReady()) return
  closeSiteDefectPopup()
  state.editingSites=!state.editingSites
  state.selectedSites=[];state.selectedSite=null;state.selected=-1
  if (state.editingSites) $('showSites').checked=true
  dragStart=null;previewBox=null
  drawCanvas()
}
$('saveSites').onclick=async()=>{
  if (!state.siteDirty || state.savingAnnotation) return
  state.savingAnnotation=true;$('annotationWorkspace').inert=true
  try{await persistSites();toast('点位模板已保存，同板型校正图片会自动使用')}
  catch(error){toast(error.message,true)}
  finally{state.savingAnnotation=false;$('annotationWorkspace').inert=false;drawCanvas()}
}
$('showSites').onchange=drawCanvas
$('siteSearch').oninput=renderSites
function siteCondition(index){
  const key=state.boxes[index]?.class
  return !key || key==='good_solder' ? '正常' : '缺陷 · '+labelOf(key)
}

function inspectSite(site){
  const extent=Math.min(.25,state.viewport.w)
  setViewport((site.x1+site.x2-extent)/2,(site.y1+site.y2-extent)/2,extent)
  state.selectedSite=site.id
  state.selected=siteBoxIndex(site)
  drawCanvas()
  toast(site.id+' · '+siteCondition(state.selected))
}

function siteOverlap(site,bounds){
  if (!bounds) return 0
  const intersection=Math.max(0,Math.min(bounds.x2,site.x2)-Math.max(bounds.x1,site.x1))
    *Math.max(0,Math.min(bounds.y2,site.y2)-Math.max(bounds.y1,site.y1))
  const union=(bounds.x2-bounds.x1)*(bounds.y2-bounds.y1)+(site.x2-site.x1)*(site.y2-site.y1)-intersection
  return union>0 ? intersection/union : 0
}

function siteBoxIndex(site,boxBounds=state.boxes.map(box=>boundsOf(displayPolygon(box)))){
  let best=-1,bestOverlap=.5
  boxBounds.forEach((bounds,index)=>{
    const overlap=siteOverlap(site,bounds)
    if (overlap>=bestOverlap){best=index;bestOverlap=overlap}
  })
  return best
}

function closeSiteDefectPopup(){
  state.sitePopup=null
  $('siteDefectPopup').hidden=true
}

function openSiteDefectPopup(site,siteIds=[site.id]){
  const sites=availableSites().filter(item=>siteIds.includes(item.id))
  if (state.editingSites || state.savingAnnotation || !sites.length) return
  state.selectedSite=site.id
  state.selected=siteBoxIndex(site)
  drawCanvas()
  const classes=sites.map(item=>state.boxes[siteBoxIndex(item)]?.class)
  const current=classes.every(key=>key===classes[0]) ? classes[0] : null
  state.sitePopup={siteIds:sites.map(item=>item.id),anchorSiteId:site.id,imageId:state.image.id}
  renderDefectPopup((sites.length>1 ? '已选 '+sites.length+' 个点位' : site.id)+' · 选择缺陷类型',
    current,sites.some(item=>siteBoxIndex(item)>=0),sites.length>1 ? '删除所选点位标注' : '删除该点位标注')
}

function openBoxDefectPopup(index){
  const box=state.boxes[index]
  if (!box || state.editingSites || state.savingAnnotation) return
  const anchor=boundsOf(displayPolygon(box))
  if (!anchor) return
  state.selected=index
  state.sitePopup={boxIndices:[index],anchor,imageId:state.image.id}
  drawCanvas()
  renderDefectPopup('修改缺陷标注',box.class,true,'删除该缺陷标注')
}

function renderDefectPopup(heading,current,canRemove,removeText){
  const popup=$('siteDefectPopup')
  popup.replaceChildren()
  const title=document.createElement('strong')
  title.textContent=heading
  popup.append(title)
  for (const type of state.meta.det_classes){
    const button=document.createElement('button')
    button.type='button'
    button.dataset.class=type.key
    button.textContent=type.label
    button.classList.toggle('current',type.key===current)
    button.onclick=()=>applySiteDefect(type.key)
    popup.append(button)
  }
  if (canRemove){
    const remove=document.createElement('button')
    remove.type='button';remove.className='remove';remove.textContent=removeText
    remove.onclick=()=>applySiteDefect(null)
    popup.append(remove)
  }
  const close=document.createElement('button')
  close.type='button';close.className='close';close.textContent='取消'
  close.onclick=()=>{closeSiteDefectPopup();$('labelCanvas').focus({preventScroll:true})}
  popup.append(close)
  hideHoverInfo()
  popup.hidden=false
  positionSiteDefectPopup()
  popup.scrollIntoView({block:'nearest',inline:'nearest'})
  const initialButton=popup.querySelector('button.current') || popup.querySelector('button')
  initialButton?.focus({preventScroll:true})
}

function positionSiteDefectPopup(){
  const popup=$('siteDefectPopup')
  if (!state.sitePopup || popup.hidden) return
  const site=state.sitePopup.anchor || availableSites().find(item=>item.id===state.sitePopup.anchorSiteId)
  if (!site) return closeSiteDefectPopup()
  const canvasRect=$('labelCanvas').getBoundingClientRect(),wrapRect=popup.parentElement.getBoundingClientRect()
  const v=state.viewport
  const x=canvasRect.left-wrapRect.left+((site.x1+site.x2)/2-v.x)/v.w*canvasRect.width
  const y=canvasRect.top-wrapRect.top+((site.y1+site.y2)/2-v.y)/v.h*canvasRect.height
  popup.style.left=Math.max(8,Math.min(x+12,wrapRect.width-popup.offsetWidth-8))+'px'
  popup.style.top=Math.max(8,Math.min(y+12,wrapRect.height-popup.offsetHeight-8))+'px'
}

function applySiteDefect(key){
  const target=state.sitePopup
  if (!target || target.imageId!==state.image?.id || state.savingAnnotation) return closeSiteDefectPopup()
  state.siteRenderKey=null
  if (target.boxIndices){
    if (key!==null && !state.meta.det_classes.some(type=>type.key===key)) return
    for (const index of [...target.boxIndices].sort((a,b)=>b-a)){
      if (!state.boxes[index]) continue
      if (key===null){
        state.boxes.splice(index,1);state.annotationDirty=true
      } else if (state.boxes[index].class!==key){
        state.boxes[index].class=key;state.annotationDirty=true
      }
    }
    state.selected=-1
    closeSiteDefectPopup();drawCanvas()
    $('labelCanvas').focus({preventScroll:true})
    return
  }
  const sites=availableSites().filter(item=>target.siteIds.includes(item.id))
  if (sites.length!==target.siteIds.length) return closeSiteDefectPopup()
  if (key===null){
    const indices=[...new Set(sites.map(site=>siteBoxIndex(site)).filter(index=>index>=0))].sort((a,b)=>b-a)
    for (const index of indices) state.boxes.splice(index,1)
    if (indices.length) state.annotationDirty=true
    state.selected=-1
  } else {
    if (!state.meta.det_classes.some(type=>type.key===key)) return
    const updates=sites.map(site=>({existing:siteBoxIndex(site),box:boxFromDisplay([site.x1,site.y1,site.x2,site.y2],key)}))
    if (updates.some(update=>!update.box)) return toast('所选点位超出当前图片拍摄范围',true)
    for (const {existing,box} of updates){
      if (existing>=0){
        if (state.boxes[existing].class!==key) state.annotationDirty=true
        state.boxes[existing].class=key
        state.selected=existing
      } else {
        state.boxes.push(box);state.selected=state.boxes.length-1;state.annotationDirty=true
      }
    }
  }
  closeSiteDefectPopup()
  drawCanvas()
  $('labelCanvas').focus({preventScroll:true})
}

document.addEventListener('pointerdown',event=>{
  if (!event.target.closest('#siteDefectPopup')) closeSiteDefectPopup()
})
document.addEventListener('keydown',event=>{
  if (event.key==='Escape' && state.sitePopup){
    closeSiteDefectPopup();$('labelCanvas').focus({preventScroll:true});event.preventDefault()
  }
})
window.addEventListener('beforeunload',event=>{
  if (state.annotationDirty || state.siteDirty){event.preventDefault();event.returnValue=''}
})

function renderModelUpdate(data){
  const model=data.latest_model
  $('latestModelLabel').title=''
  $('latestModelLabel').textContent=model ? '最新模型 '+model.version : '尚未生成模型'
  renderModelTestNotice(data.testing)
}

async function loadModelUpdate(){
  if (!state.project || state.modelLoading) return
  state.modelLoading=true
  try{
    renderModelUpdate(await api('/api/projects/'+state.project.id+'/model-update'))
  } finally {state.modelLoading=false}
}

async function loadImportMode(){
  const response=await fetch('/api/training-import-setting')
  if (!response.ok) return
  const setting=await response.json()
  $('importModeBadge').textContent=setting.enabled ? '识别图片自动入库已开启' : '识别图片自动入库已关闭'
}

$('importTrainingImages').onclick=()=>$('trainingFiles').click()
$('trainingFiles').onchange=async event=>{
  const files=[...event.target.files]
  event.target.value=''
  if (!files.length || !state.project || !leaveAnnotation()) return
  state.importingImages=true
  const button=$('importTrainingImages')
  button.disabled=true
  $('annotationWorkspace').inert=true
  const saved=[],failed=[]
  let duplicateCount=0
  try{
    for (const [index,file] of files.entries()){
      button.textContent='导入中 '+(index+1)+' / '+files.length
      const data=new FormData()
      data.append('files',file,file.name)
      try{
        const result=await api('/api/projects/'+state.project.id+'/images',data)
        saved.push(...result.saved)
        duplicateCount+=result.duplicate_count || 0
      }catch(error){failed.push(file.name+'：'+error.message)}
    }
    state.importingImages=false
    if (saved.length){
      state.filter='all';state.imageLimit=200
      await loadProject()
      await loadLabelImages(false)
      await openImage(saved[saved.length-1])
      await loadModelUpdate()
    }
    toast('新增 '+saved.length+' 张'+(duplicateCount ? '，已跳过 '+duplicateCount+' 张重复图片' : '')+(failed.length ? '；失败 '+failed.length+' 张：'+failed.join('；') : saved.length ? '，请标注后保存' : ''),!!failed.length)
  }catch(error){toast(error.message,true)}
  finally{
    state.importingImages=false
    button.disabled=false;button.textContent='↑ 导入图片'
    $('annotationWorkspace').inert=false
  }
}

document.querySelectorAll('[data-img-filter]').forEach(button=>{
  button.onclick=async()=>{
    if (!leaveAnnotation()) return
    state.filter=button.dataset.imgFilter
    state.imageLimit=200
    try{await loadLabelImages()}catch(error){toast(error.message,true)}
  }
})

;(async()=>{
  try{
    state.meta=await api('/api/meta')
    const project=await api('/api/inspection-workspace')
    state.project=await api('/api/projects/'+project.id)
    if (location.hash) history.replaceState(null,'',location.pathname+location.search)
    setInterval(()=>loadModelUpdate().catch(()=>{
      $('latestModelLabel').title='模型信息暂时无法刷新'
    }),3000)
    await refreshWorkspace()
    $('importTrainingImages').disabled=false
    loadImportMode().catch(()=>{})
    setInterval(()=>loadImportMode().catch(()=>{}),5000)
    setInterval(()=>refreshImageLibrary().catch(error=>toast(error.message,true)),5000)
  }catch(error){toast(error.message,true)}
})()

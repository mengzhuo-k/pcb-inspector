const $ = id => document.getElementById(id)
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))
const message = value => { $('status').textContent = value }
let stream, version = 0, localUrl, currentBoxes = [], historyRows = [], busy = false, currentRecordId = null
let currentClassification = null
let currentInspection = null
let inspectionInput = null
let currentReadOnly = false
let storedImageRequest=0
let inspectionFilename = "inspection.jpg"
const date = value => new Date(value).toLocaleString('zh-CN', {hour12:false})
const selectedTargets = new Set()
let overlaysVisible = true
const imageView = {zoom:1, x:0, y:0, brightness:100}
let imageDrag = null
let currentPage = 'work', reviewOrigin = 'work', batchOrigin = 'processing'

function updateImageView() {
  const frame = $('frame'), img = $('photo')
  if (img.naturalWidth && frame.clientWidth && frame.clientHeight) {
    const ratio = Math.min(frame.clientWidth/img.naturalWidth, frame.clientHeight/img.naturalHeight)
    const maxX = Math.max(0, (img.naturalWidth*ratio*imageView.zoom-frame.clientWidth)/2)
    const maxY = Math.max(0, (img.naturalHeight*ratio*imageView.zoom-frame.clientHeight)/2)
    imageView.x = Math.max(-maxX, Math.min(maxX, imageView.x))
    imageView.y = Math.max(-maxY, Math.min(maxY, imageView.y))
  }
  $('imageSurface').style.transform = `translate(${imageView.x}px,${imageView.y}px) scale(${imageView.zoom})`
  img.style.filter = `brightness(${imageView.brightness}%)`
  frame.classList.toggle('can-pan', imageView.zoom > 1)
  $('imageZoomValue').textContent = Math.round(imageView.zoom*100)+'%'
  $('imageZoomOut').disabled = imageView.zoom <= .5
  $('imageZoomIn').disabled = imageView.zoom >= 8
  $('imageBrightness').value = imageView.brightness
  $('imageBrightnessValue').textContent = imageView.brightness+'%'
  drawLeaders()
}
function zoomImage(factor, point) {
  const next = Math.max(.5, Math.min(8, imageView.zoom*factor))
  const anchor = point || {x:0,y:0}
  imageView.x = anchor.x-(anchor.x-imageView.x)*next/imageView.zoom
  imageView.y = anchor.y-(anchor.y-imageView.y)*next/imageView.zoom
  imageView.zoom = next
  updateImageView()
}
function fitImage() {
  imageView.zoom = 1
  imageView.x = imageView.y = 0
  updateImageView()
}
$('imageZoomIn').onclick = () => zoomImage(1.25)
$('imageZoomOut').onclick = () => zoomImage(1/1.25)
$('imageFit').onclick = fitImage
$('imageBrightness').oninput = event => {
  imageView.brightness = Number(event.target.value)
  updateImageView()
}
$('imageShowSites').onchange = draw
$('imageViewReset').onclick = () => {
  imageView.brightness = 100
  $('imageShowSites').checked = true
  overlaysVisible = true
  updateOverlayButton()
  fitImage()
  draw()
}
$('frame').addEventListener('wheel', event => {
  if (event.target.closest('.inspection-view-tools') || !event.deltaY) return
  event.preventDefault()
  const rect = $('frame').getBoundingClientRect()
  zoomImage(event.deltaY < 0 ? 1.2 : 1/1.2, {
    x:event.clientX-rect.left-rect.width/2, y:event.clientY-rect.top-rect.height/2,
  })
}, {passive:false})
$('frame').addEventListener('pointerdown', event => {
  if (event.button !== 0 || !event.isPrimary || imageView.zoom <= 1 || event.target.closest('button,input,label,.inspection-view-tools')) return
  imageDrag = {id:event.pointerId, x:event.clientX, y:event.clientY, originX:imageView.x, originY:imageView.y}
  $('frame').setPointerCapture(event.pointerId)
  $('frame').classList.add('is-panning')
})
$('frame').addEventListener('pointermove', event => {
  if (!imageDrag || imageDrag.id !== event.pointerId) return
  imageView.x = imageDrag.originX+event.clientX-imageDrag.x
  imageView.y = imageDrag.originY+event.clientY-imageDrag.y
  updateImageView()
})
function endImageDrag(event) {
  if (!imageDrag || imageDrag.id !== event.pointerId) return
  imageDrag = null
  $('frame').classList.remove('is-panning')
  if ($('frame').hasPointerCapture(event.pointerId)) $('frame').releasePointerCapture(event.pointerId)
}
for (const name of ['pointerup','pointercancel','lostpointercapture']) $('frame').addEventListener(name,endImageDrag)

function toggleTarget(index) {
  if (selectedTargets.has(index)) selectedTargets.delete(index)
  else selectedTargets.add(index)
  updateSelection()
  draw()
}

function updateSelection() {
  for (const item of $('callouts').children) {
    const active = selectedTargets.has(Number(item.dataset.index))
    item.classList.toggle('active',active)
    item.setAttribute('aria-pressed',String(active))
  }
  $('selectedCount').textContent = '已选 '+selectedTargets.size+' / '+currentBoxes.length
  $('selectAllTargets').disabled = !currentBoxes.length || selectedTargets.size === currentBoxes.length
  $('clearTargets').disabled = !selectedTargets.size
}

async function api(path, data, method = data ? 'POST' : 'GET') {
  const res = await fetch(path, {method, body:data})
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw Error(body.error || '请求失败，请检查服务连接')
  return body
}
function stopCamera() {
  version++
  stream?.getTracks().forEach(track => track.stop())
  stream = null
  $('video').srcObject = null
  $('camera').textContent = '开启实时采集'
}
function draw() {
  const img = $('photo'), layer = $('marks')
  if (!img.naturalWidth) return
  const frame = {width:$('frame').clientWidth, height:$('frame').clientHeight}
  if (!frame.width || !frame.height) return
  const ratio = Math.min(frame.width/img.naturalWidth, frame.height/img.naturalHeight)
  const width = img.naturalWidth*ratio, height = img.naturalHeight*ratio
  Object.assign(layer.style, {width:width+'px', height:height+'px',
    left:(frame.width-width)/2+'px', top:(frame.height-height)/2+'px'})
  layer.replaceChildren()
  for (const site of ($('imageShowSites').checked ? currentInspection?.sites || [] : [])){
    const [x1,y1,x2,y2]=site.rect
    const mark=document.createElement('div')
    mark.className='inspection-site'
    mark.title='检查点位 '+site.id
    Object.assign(mark.style,{left:x1*100+'%',top:y1*100+'%',width:(x2-x1)*100+'%',height:(y2-y1)*100+'%'})
    layer.append(mark)
  }
  currentBoxes.forEach((box, i) => {
    const [x,y,right,bottom] = box.rect
    const active = selectedTargets.has(i)
    const mark = document.createElement('button')
    mark.type = 'button'
    mark.setAttribute('aria-label',box.label+' #'+(i+1))
    mark.title='缺陷：'+box.label+' · '+(box.score*100).toFixed(1)+'%'
    mark.setAttribute('aria-pressed',String(active))
    mark.onclick = () => toggleTarget(i)
    mark.className = 'box'+(active ? ' selected' : '')
    mark.style.borderColor = active ? '#38bdf8' : '#f59e0b99'
    mark.style.opacity = !selectedTargets.size || active ? '1' : '.25'
    Object.assign(mark.style,{left:x*100+'%',top:y*100+'%',width:(right-x)*100+'%',height:(bottom-y)*100+'%'})
    layer.append(mark)
  })
  layer.hidden = !overlaysVisible
  updateImageView()
}
function drawLeaders() {
  const canvas = $('leaders'), viewer = $('viewer').getBoundingClientRect()
  const scale = window.devicePixelRatio || 1
  canvas.width = Math.round(viewer.width*scale)
  canvas.height = Math.round(viewer.height*scale)
  const ctx = canvas.getContext('2d')
  ctx.scale(scale,scale)
  if (!overlaysVisible || !selectedTargets.size) return
  const rail = $('callouts').getBoundingClientRect()
  const below = rail.top >= $('frame').getBoundingClientRect().bottom-1
  const boxes = $('marks').querySelectorAll('.box')
  for (const label of $('callouts').children) {
    const index = Number(label.dataset.index)
    if (!selectedTargets.has(index)) continue
    const target = boxes[index]
    if (!target) continue
    const a = target.getBoundingClientRect(), b = label.getBoundingClientRect()
    const imageFrame = $('frame').getBoundingClientRect()
    if (a.right <= imageFrame.left || a.left >= imageFrame.right || a.bottom <= imageFrame.top || a.top >= imageFrame.bottom) continue
    if (b.top < rail.top || b.bottom > rail.bottom+1) continue
    const x1 = Math.max(imageFrame.left, Math.min(imageFrame.right, below ? a.left : a.right))-viewer.left
    const y1 = Math.max(imageFrame.top, Math.min(imageFrame.bottom, (a.top+a.bottom)/2))-viewer.top
    const x2 = (below ? (b.left+b.right)/2 : b.left)-viewer.left
    const y2 = (below ? b.top : (b.top+b.bottom)/2)-viewer.top
    ctx.strokeStyle = '#38bdf8'
    ctx.fillStyle = ctx.strokeStyle
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(x1,y1)
    // Route the bend through the image margin to limit lines over board details.
    const frame = $('frame').getBoundingClientRect()
    if (below) {
      // Keep the mobile connector in the margin around the inspection form.
      ctx.lineTo(5,y1)
      ctx.lineTo(5,y2-5)
      ctx.lineTo(x2,y2-5)
    } else {
      ctx.lineTo(frame.right-viewer.left+7,y1)
      ctx.lineTo(frame.right-viewer.left+7,y2)
    }
    ctx.lineTo(x2,y2)
    ctx.stroke()
    ctx.beginPath()
    ctx.arc(x1,y1,2.5,0,Math.PI*2)
    ctx.fill()
  }
}
function renderVerdict(classification, processed) {
  const el = $('boardVerdict')
  if (!processed || !classification) {
    el.hidden = true
    el.replaceChildren()
    return
  }
  const pct = value => (100*value).toFixed(1)+'%'
  const defect = Boolean(classification.is_defect)
  const alternatives = (classification.top || []).slice(1)
    .map(item => esc(item.label)+' '+pct(item.score)).join(' · ')
  el.hidden = false
  el.className = 'board-verdict '+(defect ? 'is-defect' : 'is-good')
  el.innerHTML = '<div class="verdict-head"><span class="verdict-chip">整板识别</span><strong>'+esc(classification.label)+'</strong><span class="verdict-score">'+pct(classification.score)+'</span></div>'
    + '<p>'+(defect ? '判定不合格 · 缺陷概率 '+pct(classification.defect_score) : '未达缺陷判定阈值')
    + '（阈值 '+pct(classification.threshold)+'）</p>'
    + (alternatives ? '<small>次选：'+alternatives+'</small>' : '')
    + '<small class="verdict-note">整板分类显示缺陷类型，检测框显示定位结果。</small>'
}
function showImage(input, boxes, source, classification = null, inspection = null) {
  if (localUrl) URL.revokeObjectURL(localUrl)
  localUrl = input instanceof Blob ? URL.createObjectURL(input) : null
  currentBoxes = boxes
  currentClassification = classification
  currentInspection = inspection
  $('templateStatus').hidden=!inspection
  $('templateStatus').textContent=inspection ? '点位模板 · 已检查 '+inspection.checked_sites+' / '+inspection.total_sites+' 处' : ''
  selectedTargets.clear()
  const processed = source === '已建档' || source.startsWith('实时检测') || source.startsWith('智能检测')
  document.querySelector('.detection-panel').classList.toggle('has-results',processed)
  const defectCount = inspection ? new Set(boxes.map(box=>box.site_id)).size : boxes.length
  $('detectionCount').textContent = processed ? (inspection ? defectCount+' / '+inspection.checked_sites+' 处检出缺陷' : boxes.length+' 个点位') : '待检测'
  $('detectionEmpty').hidden = !!boxes.length || !!classification
  $('detectionEmpty').innerHTML = boxes.length ? '' : '未检出超过阈值的缺陷点位<br><small>整板分类结果显示在上方</small>'
  if (inspection && !boxes.length) $('detectionEmpty').textContent='已检查 '+inspection.checked_sites+' 处模板点位，未检出超过阈值的缺陷。'
  renderVerdict(classification, processed)
  $('callouts').replaceChildren()
  boxes.forEach((box,i)=>{
    const label = document.createElement('button')
    label.type = 'button'
    label.className = 'callout'
    label.dataset.index = i
    label.setAttribute('aria-pressed','false')
    label.innerHTML = '<span class="number">'+esc(box.site_id || '#'+String(i+1).padStart(2,'0'))+'</span><strong>'+esc(box.label)+'</strong><span class="score">'+(box.score*100).toFixed(1)+'<small>%</small></span><span class="confidence-label">识别置信度</span>'
    label.onclick = () => toggleTarget(i)
    $('callouts').append(label)
  })
  updateSelection()
  $('marks').replaceChildren()
  $('photo').src = localUrl || input
  $('source').textContent = source
  $('leaders').getContext('2d').clearRect(0,0,$('leaders').width,$('leaders').height)
  if (!boxes.length && !processed) {
    $('detectionEmpty').innerHTML = '载入图像后开始检测<br><small>序号与置信度显示在此处</small>'
  }
}
$('photo').onload = draw
new ResizeObserver(draw).observe($('frame'))
$('callouts').addEventListener('scroll',drawLeaders)
$('selectAllTargets').onclick = () => {
  currentBoxes.forEach((box,index)=>selectedTargets.add(index))
  updateSelection()
  draw()
}
$('clearTargets').onclick = () => {
  selectedTargets.clear()
  updateSelection()
  draw()
}
function updateOverlayButton() {
  $('overlays').setAttribute('aria-pressed',String(overlaysVisible))
  $('overlays').textContent = overlaysVisible ? '标注已开启' : '标注已关闭'
}
$('overlays').onclick = () => {
  overlaysVisible = !overlaysVisible
  updateOverlayButton()
  draw()
}
$('expand').onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen()
    else await $('viewer').requestFullscreen()
  } catch (err) { message('当前浏览器无法全屏显示：'+err.message) }
}
document.addEventListener('fullscreenchange',draw)
function resetCaptureView(){
  currentRecordId = null
  currentReadOnly=false
  $('reviewRecord').hidden=true
  $('inspect').inert=false
  $('run').hidden=false
  $('run').disabled=false
  document.querySelector('#work .toolbar').hidden=false
  reviewOrigin='work'
  $('backToBatch').hidden=true
  $('previousImage').hidden=true;$('nextImage').hidden=true
  $('title').textContent='采集检测'
  $('pageDescription').textContent='采集板面图像，自动分析缺陷并生成检验单'
  inspectionInput=null
}

async function selectImage(input, source, options={}) {
  stopCamera()
  ++storedImageRequest
  const token=version
  resetCaptureView()
  inspectionInput = input
  inspectionFilename=input instanceof File ? input.name : 'inspection.jpg'
  $('viewingFile').textContent=input instanceof File ? input.name : ''
  $('alignmentStatus').hidden=true
  showImage(input, [], source, null)
  $('result').replaceChildren()
  message('正在矫正板面并识别缺陷…')
  $('run').disabled=true
  try{
    const blob=input instanceof Blob ? input : await (await fetch(input)).blob()
    const form=new FormData()
    form.append('image',blob,inspectionFilename)
    form.append('station',$('inspect').elements.station.value)
    form.append('batch',$('inspect').elements.batch.value)
    if (options.previewOnly) form.append('preview_only','1')
    const result=await api('/api/preview',form)
    if (token!==version || currentRecordId) return
    showImage(result.image || blob,result.boxes,'智能检测 · '+result.boxes.length+' 项缺陷',result.classification,result.inspection)
    showAlignmentStatus(result.alignment)
    message(result.issues?.length ? result.issues.join('；') : '识别完成，可核对结果并生成检验单。')
  }catch(error){if(token===version) message(error.message)}
  finally{if(token===version) $('run').disabled=false}
}

async function inspectStoredRecord(id,origin,ids){
  if (busy) return message('请等待当前检测完成')
  const requestId=++storedImageRequest
  try{
    const record=await api((origin==='rejected' ? '/api/exceptions/' : '/api/records/')+id)
    if (requestId!==storedImageRequest) return
    if (ids) reviewIds=ids
    reviewOrigin=origin
    switchPage('work')
    present(record)
    message('已载入保存的板面图片和识别结果。')
    $('viewer').scrollIntoView({behavior:'smooth',block:'start'})
  }catch(error){if(requestId===storedImageRequest) message(error.message)}
}

for (const id of ['file','snap']) $(id).onchange = e => {
  const file = e.target.files[0]
  if (file) selectImage(file,'已载入图像')
  e.target.value = ''
}
$('sample').onclick = () => selectImage('/static/sample-missing.png?v=20261110b','漏焊样例')
async function capture() {
  const video = $('video'), canvas = document.createElement('canvas')
  if (!video.videoWidth) throw Error('正在等待摄像头画面')
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  canvas.getContext('2d').drawImage(video,0,0)
  return new Promise(resolve => canvas.toBlob(resolve,'image/jpeg',.94))
}
$('camera').onclick = async () => {
  if (stream) { stopCamera(); message('采集已停止，已保留最后一帧。'); return }
  const token = ++version
  try {
    if (!navigator.mediaDevices) throw Error('摄像头需要 HTTPS 或本机 localhost 访问')
    const media = await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1280}},audio:false})
    if (version !== token) {media.getTracks().forEach(t=>t.stop()); return}
    stream = media
    $('video').srcObject = media
    await $('video').play()
    ++storedImageRequest
    resetCaptureView()
    inspectionFilename='frame.jpg'
    $('viewingFile').textContent='实时采集'
    $('result').replaceChildren()
    $('camera').textContent = '停止采集'
    message('正在逐帧检测，点击建档可保存一次检验。')
    while (stream && version === token) {
      const blob = await capture(), form = new FormData()
      form.append('image',blob,'frame.jpg')
      form.append('live_preview','1')
      form.append('station', $('inspect').elements.station.value)
      const result = await api('/api/preview',form)
      if (token === version){
        inspectionInput=blob
        showImage(result.image || blob,result.boxes,'实时检测 · '+result.boxes.length+' 项缺陷',result.classification,result.inspection)
        showAlignmentStatus(result.alignment)
      }
      await new Promise(resolve => setTimeout(resolve,350))
    }
  } catch (err) { if (token === version) {stopCamera();message('采集失败：'+err.message)} }
}
function showAlignmentStatus(alignment){
  $('alignmentStatus').hidden=!alignment
  $('alignmentStatus').textContent=alignment ? '板面已矫正' : ''
}

function present(r) {
  $('title').textContent='采集检测'
  $('pageDescription').textContent='已保存的识别结果 · '+(r.readonly ? '归档结果仅供查看' : '员工可自主选择复核')
  stopCamera()
  currentRecordId = r.id
  currentReadOnly=!!r.readonly || !!r.exception_mode
  $('inspect').inert=true
  $('run').hidden=true
  document.querySelector('#work .toolbar').hidden=false
  inspectionInput=null
  inspectionFilename=r.filename || 'inspection.jpg'
  showAlignmentStatus(r.alignment)
  $('viewingFile').textContent = r.filename || ''
  if (typeof updateReviewNavigation === 'function') updateReviewNavigation(r)
  $('backToBatch').hidden = false
  $('backToBatch').textContent = reviewOrigin === 'rejected' ? '← 返回不合格产品' : (reviewOrigin === 'batch' && r.batch_id ? '← 返回批次结果' : '← 返回检验记录')
  $('backToBatch').onclick = async () => {
    const target = reviewOrigin === 'batch' && r.batch_id ? 'batch' : reviewOrigin === 'rejected' ? 'rejected' : 'logs'
    switchPage(target)
    try {
      if (target === 'batch') await openBatch(r.batch_id)
      else await refreshWorkspace()
    } catch(err) { message(err.message) }
  }
  showImage('/api/records/'+r.id+'/image',r.boxes,'已建档',r.classification,r.inspection)
  for (const key of ['station','batch','note']) $('inspect').elements[key].value = r[key] ?? ''
  $('stationName').textContent = r.station
  updateRate()
  const findings = r.analysis.filter(a=>a.label!=='未见明显缺陷').map(a=>'<div class="finding"><h3>'+esc(a.label)+'</h3><p>可能根因 · '+esc(a.cause)+'</p><p>改进建议 · '+esc(a.action)+'</p><small>'+esc(a.evidence)+'</small></div>').join('')
  const sheet=[
    ['检验编号',r.id.slice(0,12)],['采集时间',date(r.created)],['生产工位',r.station],['生产批次',r.batch],
    ['检测模型',r.model_version || '历史版本'],
    ...(r.classification?.authoritative ? [['判定依据','整板分类模型'],['点位辅助模型',r.classification.point_model_version || '未启用']] : []),
  ].map(([name,value])=>'<div><dt>'+esc(name)+'</dt><dd>'+esc(value)+'</dd></div>').join('')
  const audit = r.reviews?.length ? '<details class="review-history"><summary>复查记录（'+r.reviews.length+' 次）</summary>'+r.reviews.map(item=>'<p>'+date(item.created)+'　'+esc(item.reviewer)+'　'+esc(item.decision)+'　'+esc(item.comment)+'</p>').join('')+'</details>' : ''
  const version = r.model_version || r.boxes[0]?.model || '历史记录未标明版本'
  const classification = r.classification
  const verdict = classification
    ? '<div class="result-verdict '+(classification.is_defect ? 'is-defect' : 'is-good')+'"><strong>整板识别 · '+esc(classification.label)+' '+Math.round(classification.score*100)+'%</strong><span>缺陷概率 '+Math.round(classification.defect_score*100)+'%（阈值 '+Math.round(classification.threshold*100)+'%）· '+(classification.is_defect ? '初判不合格' : '未超过分类阈值')+'</span></div>'
    : '<p class="muted">本单生成时未启用整板缺陷分类。</p>'
  const manualOptions = ['漏焊','虚焊','冷焊','桥接','多锡','焊盘损伤','元件偏移','缺件','氧化','烧伤']
  const manualChecked = Array.isArray(r.manual) ? r.manual : []
  const manualTags = '<div class="manual-tags"><span>补标缺陷类型（可多选）</span>'+manualOptions.map(item=>'<label><input type="checkbox" name="manual" value="'+item+'"'+(manualChecked.includes(item) ? ' checked' : '')+'>'+item+'</label>').join('')+'</div>'
  const review = r.review_available ? '<details class="optional-review"><summary>自主复核</summary><p class="muted">默认采用机器判定；需要调整时展开复核，不复核也可归档。</p><form id="review"><h3>'+(r.signed ? '复核已签字 · '+esc(r.reviewer) : '复核签字')+'</h3><p class="muted">机器初判 '+esc(r.auto_decision || r.decision)+' · '+(r.signed ? '签字时间 '+date(r.signed) : '如需复核，请核对缺陷位置和检验结论后签字')+'</p><input type="hidden" name="revision" value="'+(r.revision || 0)+'"><details><summary>补充缺陷与复核意见</summary>'+manualTags+'<label>复核意见<textarea name="comment" maxlength="2000" rows="2">'+esc(r.comment)+'</textarea></label></details><label>复核结论<select name="decision" required><option value="合格">合格</option><option value="不合格">不合格</option></select></label><label>质检员签字<input name="reviewer" maxlength="80" required value="'+esc(r.reviewer || localStorage.getItem('pcbReviewer') || '')+'" placeholder="输入姓名"></label><button class="primary">确认结论并签字</button></form></details>'+audit : audit
  $('result').innerHTML='<article><div class="cardhead"><h2>'+(r.exception_mode ? '不合格图片 · 归档只读' : r.readonly ? '归档检验单 · 只读' : '自动检验单')+'</h2><div class="viewactions"><span class="tag" data-state="'+esc(r.decision || '待检测')+'">'+esc(r.decision || '待检测')+'</span></div></div><div class="body"><dl class="inspection-sheet">'+sheet+'</dl>'+verdict+'<h3>缺陷分析与改进建议 · '+r.boxes.length+' 个定位结果</h3><div class="findings-grid">'+(findings || '<p class="muted">当前模型未检出超过阈值的缺陷，可按需结合板面图像进行复核。</p>')+'</div>'+(r.note ? '<p class="record-note">现场记录 · '+esc(r.note)+'</p>' : '')+'</div>'+review+'</article>'
  if (r.issues?.length) {
    const warning = document.createElement('p')
    warning.className = 'hint'
    warning.textContent = '检测提示 · ' + r.issues.join('；')
    $('result').prepend(warning)
  }
  $('reviewRecord').hidden=!r.review_available
  $('reviewRecord').textContent=r.signed ? '重新复核（可选）' : '复核（可选）'
  $('reviewRecord').onclick=()=>{
    const panel=document.querySelector('.optional-review')
    if (!panel) return
    panel.open=true
    panel.scrollIntoView({behavior:'smooth',block:'start'})
    $('review').elements.decision.focus({preventScroll:true})
  }
  if ($('review')) $('review').elements.decision.value = r.decision || ''
  if ($('review')) $('review').onsubmit = async e => {
    e.preventDefault()
    const btn = e.submitter
    localStorage.setItem('pcbReviewer',e.target.elements.reviewer.value.trim())
    btn.disabled = true
    try {
      const updated=await api('/api/records/'+r.id+'/review',new FormData(e.target))
      present(updated)
      const updates=await Promise.allSettled([refresh(),refreshWorkspace(),...(updated.batch_id ? [openBatch(updated.batch_id)] : [])])
      message(updates.some(result=>result.status==='rejected') ? '复核已保存，部分列表暂时未能刷新，请重新打开批次查看。' : '复核已保存，检测任务中的图片结论、签字信息和批次统计已同步更新。')
    }
    catch(err){message(err.message)} finally {btn.disabled=false}
  }
}
$('inspect').onsubmit = async e => {
  e.preventDefault()
  if (busy || currentReadOnly || currentRecordId) return
  busy = true
  for (const id of ['run','camera','file','snap','sample']) $(id).disabled=true
  $('run').textContent='正在检测…'
  try {
    const data = new FormData(e.target)
    const input=inspectionInput || $('photo').src
    const blob = stream ? await capture() : input instanceof Blob ? input : await (await fetch(input)).blob()
    stopCamera()
    data.append('image',blob,inspectionFilename)
    const r = await api('/api/records',data)
    present(r)
    await refresh()
    message('检验单已自动生成，默认采用机器判定，可自主选择复核。')
    $('result').scrollIntoView({behavior:'smooth'})
  } catch(err) {message(err.message)} finally {
    busy=false
    for (const id of ['run','camera','file','snap','sample']) $(id).disabled=false
    $('run').textContent='检测并生成检验单'
  }
}
function recordRow(r) {
  const row = document.createElement('div')
  row.className='historyrow'
  const state = r.decision || '待检测'
  row.innerHTML='<strong>'+esc(r.filename || r.batch)+'</strong><span>'+esc(r.station)+'</span><span>'+date(r.created)+'</span><span>'+r.count+' 项候选</span><span class="tag" data-state="'+esc(state)+'">'+esc(state)+'<br><small>'+(r.signed ? '已复查' : r.sampled ? '抽样未复核（可选）' : '机器判定')+'</small></span><div class="rowactions"><button class="open-record navigation-button">查看检验单 →</button></div>'
  row.querySelector('.open-record').onclick=()=>inspectStoredRecord(r.id,'logs')
  return row
}
function isPatrolRecord(record) {
  return !record.batch_id && /^(巡检|巡视)-/.test(record.batch || '')
}
function renderHistory() {
  if (typeof renderBatchHistory === 'function') renderBatchHistory()
}
async function refresh() {
  historyRows=await api('/api/records')
  updateRate()
  renderHistory()
  renderStatistics()
}
function renderStatistics() {
  const days = Array.from({length:7},(_,i)=>{
    const day = new Date()
    day.setDate(day.getDate()-6+i)
    const count = historyRows.filter(row=>!isPatrolRecord(row) && new Date(row.created).toDateString() === day.toDateString()).length
    return {label:(day.getMonth()+1)+'/'+day.getDate(), count}
  })
  const max = Math.max(1,...days.map(day=>day.count))
  $('trend').innerHTML = days.map(day=>'<div class="trend-day" aria-label="'+day.label+' '+day.count+' 份检验单"><b>'+day.count+'</b><span class="trend-bar" style="height:'+Math.round(day.count/max*65)+'%"></span><span>'+day.label+'</span></div>').join('')
}
function updateRate() {
  const batch = $('inspect').elements.batch.value.trim()
  const signed = historyRows.filter(row => row.batch === batch && ['合格','不合格'].includes(row.decision))
  const passed = signed.filter(row => row.decision === '合格').length
  $('passRate').textContent = signed.length ? (100*passed/signed.length).toFixed(1)+'%' : '—'
  $('rateBasis').textContent = signed.length ? passed+' 份合格 / '+signed.length+' 份已判定' : (batch ? '该批次暂无已判定检验单' : '输入批次后显示')
}
$('inspect').elements.batch.addEventListener('input',updateRate)
$('inspect').elements.station.addEventListener('input',e=>{$('stationName').textContent=e.target.value.trim() || '—'})
function switchPage(page) {
  if (busy) return
  stopCamera()
  currentPage = page
  for (const name of ['import','processing','rejected','logs','batch','work','training']) $(name).hidden = name !== page
  const descriptions = {
    import:['批量导入','填写工位与批次，导入生产图片后自动检测并建档'],
    processing:['检测批次','打开批次查看进度、处理失败项或安排抽样复查'],
    rejected:['不合格产品','仅保存已完成批次的不合格图片，按批次、序号和文件名查看归档'],
    logs:['质量档案','追溯采集图像、分析结果与签字记录'],
    batch:['批次处理','查看检测进度、逐图结果和抽样清单'],
    work:['采集检测','采集板面图像，自动分析缺陷并生成检验单'],
    training:['模型训练','上传并标注样本，训练完成后自动更新检测模型'],
  }
  $('title').textContent=descriptions[page][0]
  $('pageDescription').textContent=descriptions[page][1]
  if (page === 'training' && !$('trainingFrame').getAttribute('src')) $('trainingFrame').src=$('trainingFrame').dataset.src
  const owner = page === 'batch' ? batchOrigin : page
  document.querySelectorAll('.nav[data-page]').forEach(n=>n.classList.toggle('active',n.dataset.page===owner))
  const url=new URL(location.href)
  url.searchParams.set('page',page==='batch' ? batchOrigin : page)
  history.replaceState(null,'',url)
  message('')
}
document.querySelectorAll('.nav[data-page]').forEach(n=>n.onclick=()=>{
  switchPage(n.dataset.page)
  refresh().catch(err=>message(err.message))
  refreshWorkspace().catch(err=>message(err.message))
})
$('filter').oninput=renderHistory
$('voice').onclick=()=>{
  const Speech=window.SpeechRecognition || window.webkitSpeechRecognition
  if (!Speech) {message('当前浏览器未提供语音识别，可使用手机键盘的语音输入。');return}
  const speech=new Speech()
  speech.lang='zh-CN'
  speech.onresult=e=>{const note=$('inspect').elements.note;note.value=(note.value+' '+e.results[0][0].transcript).trim().slice(0,2000);message('语音已转写，请核对现场备注。')}
  speech.onerror=e=>message('语音识别失败：'+e.error)
  speech.onend=()=>{$('voice').disabled=false;$('voice').textContent='语音录入'}
  try{speech.start()}catch(error){message('无法启动语音录入：'+error.message);return}
  $('voice').disabled=true
  $('voice').textContent='正在聆听…'
}
window.addEventListener('pagehide',stopCamera)
refresh().catch(err=>message(err.message))

const requestedPage=new URLSearchParams(location.search).get('page')
switchPage(['work','import','processing','rejected','logs'].includes(requestedPage) ? requestedPage : 'work')

let importSettingSaving=false
async function loadTrainingImportSetting(){
  if (importSettingSaving) return
  try{
    const setting=await api('/api/training-import-setting')
    if (importSettingSaving) return
    $('trainingImportSwitch').checked=setting.enabled
    $('trainingImportSwitch').disabled=false
    $('trainingImportState').textContent=setting.enabled ? '已开启' : '已关闭'
  }catch(error){$('trainingImportState').textContent='读取失败';$('trainingImportSwitch').title=error.message}
}
$('trainingImportSwitch').onchange=async event=>{
  const enabled=event.target.checked
  importSettingSaving=true;event.target.disabled=true
  $('trainingImportState').textContent='保存中'
  try{
    const response=await fetch('/api/training-import-setting',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled})})
    const data=await response.json()
    if (!response.ok) throw Error(data.error || '设置保存失败')
    event.target.checked=data.enabled
    $('trainingImportState').textContent=data.enabled ? '已开启' : '已关闭'
    message(data.enabled ? '图片自动入库已开启，后续导入的图片将进入训练图库。' : '图片自动入库已关闭，后续导入的图片仅用于识别。')
  }catch(error){event.target.checked=!enabled;message(error.message);$('trainingImportState').textContent=event.target.checked ? '已开启' : '已关闭'}
  finally{importSettingSaving=false;event.target.disabled=false}
}
loadTrainingImportSetting()
setInterval(loadTrainingImportSetting,5000)

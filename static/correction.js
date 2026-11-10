const correctionClasses = {
  missing_connection:'漏焊/缺焊', poor_solder_virtual:'虚焊', poor_solder_cold:'冷焊',
  solder_bridge:'桥接', pad_damage:'焊盘损伤', component_shift:'元件偏移', missing_component:'缺件'
}
let correctionCanvas = null
function closeCorrection() {
  correctionCanvas?.remove()
  correctionCanvas = null
}
function installCorrection(record) {
  const section = document.createElement('section')
  section.innerHTML = '<button type="button" id="editCorrection">纠错并转训练样本</button><div id="correctionTools" hidden></div>'
  $('result').append(section)
  $('editCorrection').onclick = async () => {
    try {
      const projects = await api('/platform/api/projects')
      const tools = $('correctionTools')
      tools.hidden = false
      tools.innerHTML = '<p>拖动画框补充漏检，点击框后可修改类别或删除误报。原预测保留在检验单中。</p>'
        + '<label>缺陷类别<select id="correctionClass">'+Object.entries(correctionClasses).map(([key,label])=>'<option value="'+key+'">'+label+'</option>').join('')+'</select></label>'
        + '<button type="button" id="deleteCorrection">删除选中框</button><button type="button" id="closeCorrection">结束画框</button>'
        + '<label>训练项目<select id="correctionProject">'+projects.map(p=>'<option value="'+esc(p.id)+'">'+esc(p.name)+' · '+esc(p.station)+'</option>').join('')+'</select></label>'
        + '<label>纠错人<input id="correctionReviewer" maxlength="80" value="'+esc(record.reviewer || '')+'"></label>'
        + '<button type="button" id="saveCorrection">保存纠错到训练图库</button><p id="correctionStatus" role="status"></p>'
      let boxes = structuredClone(record.corrections?.boxes || record.boxes)
      let selected = -1, start = null
      closeCorrection()
      const canvas = document.createElement('canvas')
      correctionCanvas = canvas
      Object.assign(canvas.style, {position:'absolute', zIndex:20, touchAction:'none', cursor:'crosshair'})
      $('frame').append(canvas)
      function paint() {
        const frame = $('frame').getBoundingClientRect(), img = $('photo')
        const ratio = Math.min(frame.width/img.naturalWidth,frame.height/img.naturalHeight)
        canvas.width = img.naturalWidth*ratio
        canvas.height = img.naturalHeight*ratio
        Object.assign(canvas.style,{width:canvas.width+'px',height:canvas.height+'px',left:(frame.width-canvas.width)/2+'px',top:(frame.height-canvas.height)/2+'px'})
        const ctx = canvas.getContext('2d')
        boxes.forEach((box,index)=>{
          const [x,y,right,bottom] = box.rect
          ctx.strokeStyle = index === selected ? '#38bdf8' : '#e11d48'
          ctx.lineWidth = 3
          ctx.strokeRect(x*canvas.width,y*canvas.height,(right-x)*canvas.width,(bottom-y)*canvas.height)
          ctx.fillStyle = '#ffffff'
          ctx.fillText((index+1)+' '+(correctionClasses[box.key] || box.key),x*canvas.width,y*canvas.height+12)
        })
      }
      function point(event) {
        const r = canvas.getBoundingClientRect()
        return [Math.max(0,Math.min(1,(event.clientX-r.left)/r.width)), Math.max(0,Math.min(1,(event.clientY-r.top)/r.height))]
      }
      canvas.onpointerdown = event => {start=point(event); canvas.setPointerCapture(event.pointerId)}
      canvas.onpointerup = event => {
        if (!start) return
        const end = point(event)
        if (Math.abs(end[0]-start[0])>.005 && Math.abs(end[1]-start[1])>.005) {
          boxes.push({key:$('correctionClass').value,rect:[Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.max(start[0],end[0]),Math.max(start[1],end[1])]})
          selected=boxes.length-1
        } else {
          selected=boxes.findLastIndex(box=>start[0]>=box.rect[0] && start[0]<=box.rect[2] && start[1]>=box.rect[1] && start[1]<=box.rect[3])
          if (selected>=0 && correctionClasses[boxes[selected].key]) $('correctionClass').value=boxes[selected].key
        }
        start=null
        paint()
      }
      canvas.onpointercancel = () => {start=null}
      $('correctionClass').onchange = () => {if(selected>=0) {boxes[selected].key=$('correctionClass').value;paint()}}
      $('deleteCorrection').onclick = () => {if(selected>=0) boxes.splice(selected,1);selected=-1;paint()}
      $('closeCorrection').onclick = closeCorrection
      $('saveCorrection').onclick = async event => {
        event.target.disabled=true
        try {
          const response = await fetch('/api/records/'+record.id+'/corrections',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({revision:record.revision,project_id:$('correctionProject').value,reviewer:$('correctionReviewer').value,boxes,previous_corrections:record.corrections || null})})
          const result = await response.json()
          if (!response.ok) throw Error(result.error || '保存失败')
          record.corrections={boxes:structuredClone(boxes.map(box=>({key:box.key,rect:box.rect}))),reviewer:$('correctionReviewer').value.trim(),project_id:$('correctionProject').value}
          $('correctionStatus').textContent='纠错已保存，样本已加入训练图库。'
          closeCorrection()
        } catch(error) {$('correctionStatus').textContent=error.message}
        finally {event.target.disabled=false}
      }
      paint()
    } catch(error) {message(error.message)}
  }
}

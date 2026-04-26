const initial = window.__SHOWROOM_INITIAL__
const catalog = window.__SHOWROOM_CATALOG__ ?? []
const root = document.getElementById('showroom')
const SPEEDS = [0.5, 1, 2]
const SCALES = [1, 2, 3, 4]

const escapeHtml = value =>
  String(value ?? '').replace(
    /[&<>\"']/g,
    char => ({ '&': '&amp;', '"': '&quot;', "'": '&#39;', '<': '&lt;', '>': '&gt;' })[char]
  )

// Minimal ANSI-to-HTML: handles ESC[NC (cursor forward), strips control sequences.
// No color SGR support needed — all styling comes from the Ink renderer's own output.
const ansiToHtml = raw => {
  let out = ''
  let i = 0

  while (i < raw.length) {
    if (raw[i] === '\x1b' && raw[i + 1] === '[') {
      let j = i + 2

      while (j < raw.length && raw[j] >= '0' && raw[j] <= '9') j++
      if (j < raw.length && raw[j] === ';') j++
      while (j < raw.length && raw[j] >= '0' && raw[j] <= '9') j++

      if (j < raw.length) {
        const cmd = raw[j]

        if (cmd === 'C') {
          const n = parseInt(raw.slice(i + 2, j), 10) || 1
          out += ' '.repeat(n)
        }

        // All other sequences (cursor hide/show, bracketed paste, etc.) — strip
        i = j + 1
        continue
      }
    }

    if (raw[i] === '\r' && raw[i + 1] === '\n') {
      out += '\n'
      i += 2
      continue
    }

    if (raw[i] === '\r') {
      out += '\n'
      i++
      continue
    }

    out += raw[i]
    i++
  }

  return out
}

const state = {
  body: null,
  composer: null,
  frameMode: false,
  frameTargets: new Map(),
  overlays: null,
  progressFill: null,
  progressLabel: null,
  raf: null,
  scale: 2,
  shell: null,
  speed: 1,
  startedAt: 0,
  statusLeft: null,
  statusRight: null,
  timers: [],
  total: 0,
  viewport: null,
  workflow: initial?.workflow ?? { timeline: [] }
}

const clearTimers = () => {
  while (state.timers.length) {
    clearTimeout(state.timers.pop())
  }

  if (state.raf) {
    cancelAnimationFrame(state.raf)
    state.raf = null
  }
}

const resolveTarget = id => {
  if (!id) {
    return null
  }

  return state.frameTargets.get(id) ?? document.querySelector(`[data-target="${CSS.escape(id)}"]`)
}

const setText = (node, text = '', duration = 0) => {
  if (!duration || state.speed <= 0) {
    node.textContent = text

    return
  }

  const chars = [...text]
  const adjusted = duration / state.speed
  const started = performance.now()

  const frame = now => {
    const n = Math.min(chars.length, Math.ceil(((now - started) / adjusted) * chars.length))
    node.textContent = chars.slice(0, n).join('')

    if (n < chars.length) {
      requestAnimationFrame(frame)
    }
  }

  requestAnimationFrame(frame)
}

const removeAfter = (node, duration = 1400) => {
  const wait = duration / state.speed

  state.timers.push(
    setTimeout(() => {
      node.classList.remove('is-visible')
      state.timers.push(setTimeout(() => node.remove(), 420 / state.speed))
    }, wait)
  )
}

const rectFor = (id, pad = 8) => {
  const el = resolveTarget(id)

  if (!el || !state.overlays) {
    return null
  }

  const stage = state.overlays.getBoundingClientRect()
  const rect = el.getBoundingClientRect()

  return {
    height: rect.height + pad * 2,
    left: rect.left - stage.left - pad,
    top: rect.top - stage.top - pad,
    width: rect.width + pad * 2
  }
}

const placeNear = (node, id, position = 'right') => {
  const rect = rectFor(id, 0)

  if (!rect) {
    node.style.left = '24px'
    node.style.top = '24px'

    return
  }

  const gap = 18
  const left = position === 'left' ? rect.left - node.offsetWidth - gap : rect.left + rect.width + gap
  const top = position === 'top' ? rect.top - node.offsetHeight - gap : rect.top

  node.style.left = `${Math.max(12, left)}px`
  node.style.top = `${Math.max(12, top)}px`
}

const message = action => {
  if (state.frameMode) {
    return
  }

  const roleSpec = {
    assistant: { copy: '#fff8dc', glyph: '┊', tone: '#cd7f32' },
    system: { copy: '#cc9b1f', glyph: '·', tone: '#cc9b1f' },
    tool: { copy: '#cc9b1f', glyph: '⚡', tone: '#cd7f32' },
    user: { copy: '#daa520', glyph: '❯', tone: '#ffd700' }
  }
  const spec = roleSpec[action.role] ?? roleSpec.assistant
  const line = document.createElement('div')
  const glyph = document.createElement('span')
  const copy = document.createElement('div')

  line.className = `showroom-line showroom-line-${action.role ?? 'assistant'}`
  line.dataset.target = action.id ?? ''
  line.style.setProperty('--role', spec.tone)
  line.style.setProperty('--copy', spec.copy)

  glyph.className = 'showroom-glyph'
  glyph.textContent = spec.glyph

  copy.className = 'showroom-copy'

  line.append(glyph, copy)
  state.body.append(line)
  setText(copy, action.text, action.duration)
}

const tool = action => {
  if (state.frameMode) {
    return
  }

  const box = document.createElement('div')
  const title = document.createElement('div')
  const items = document.createElement('div')

  box.className = 'showroom-tool'
  box.dataset.target = action.id ?? ''

  title.className = 'showroom-tool-title'
  title.textContent = action.title ?? 'tool activity'

  items.className = 'showroom-tool-items'

  for (const item of action.items ?? []) {
    const row = document.createElement('div')

    row.textContent = item
    items.append(row)
  }

  box.append(title, items)
  state.body.append(box)
}

const frame = action => {
  if (!action.ansi) {
    return
  }

  const pre = document.createElement('pre')
  pre.className = 'showroom-frame'
  pre.dataset.target = action.id ?? ''
  pre.innerHTML = escapeHtml(ansiToHtml(action.ansi))
  state.body.append(pre)

  if (action.id) {
    state.frameTargets.set(action.id, pre)
  }
}

const fade = action => {
  const el = resolveTarget(action.target)

  if (!el) {
    return
  }

  el.style.transition = `opacity ${(action.duration ?? 420) / state.speed}ms var(--ease-in-out)`
  el.style.opacity = String(action.to ?? 0)
}

const highlight = action => {
  const el = resolveTarget(action.target)

  if (!el) {
    return
  }

  el.classList.add('is-highlighted')
  state.timers.push(setTimeout(() => el.classList.remove('is-highlighted'), (action.duration ?? 1200) / state.speed))
}

const caption = action => {
  const node = document.createElement('div')

  node.className = 'showroom-caption'
  node.dataset.target = action.id ?? ''
  node.textContent = action.text ?? ''
  state.overlays.append(node)
  placeNear(node, action.target, action.position)
  requestAnimationFrame(() => node.classList.add('is-visible'))
  removeAfter(node, action.duration ?? 1600)
}

const spotlight = action => {
  const rect = rectFor(action.target, action.pad ?? 6)

  if (!rect) {
    return
  }

  const node = document.createElement('div')

  node.className = 'showroom-spotlight'
  node.style.left = `${rect.left}px`
  node.style.top = `${rect.top}px`
  node.style.width = `${rect.width}px`
  node.style.height = `${rect.height}px`
  state.overlays.append(node)
  requestAnimationFrame(() => node.classList.add('is-visible'))
  removeAfter(node, action.duration ?? 1500)
}

const status = action => {
  state.statusLeft.textContent = action.text ?? ''
  state.statusRight.textContent = action.detail ?? ''
}

const compose = action => setText(state.composer, action.text ?? '', action.duration ?? 0)

const clearTranscript = () => {
  state.overlays.textContent = ''
  state.frameTargets.clear()
  state.body.textContent = ''
}

const ACTIONS = { caption, clear: clearTranscript, compose, fade, frame, highlight, message, spotlight, status, tool }

const fmtTime = ms => {
  if (!Number.isFinite(ms)) {
    return '0.0s'
  }

  return `${(Math.max(0, ms) / 1000).toFixed(1)}s`
}

const tickProgress = () => {
  if (!state.startedAt) {
    return
  }

  const elapsed = Math.min(state.total, (performance.now() - state.startedAt) * state.speed)
  const ratio = state.total ? elapsed / state.total : 0

  state.progressFill.style.width = `${(ratio * 100).toFixed(2)}%`
  state.progressLabel.textContent = `${fmtTime(elapsed)} / ${fmtTime(state.total)}`

  if (elapsed < state.total) {
    state.raf = requestAnimationFrame(tickProgress)
  }
}

const play = () => {
  clearTimers()
  clearTranscript()
  state.statusLeft.textContent = ''
  state.statusRight.textContent = ''
  state.composer.textContent = state.workflow.composer ?? ''

  const timeline = [...(state.workflow.timeline ?? [])].sort((a, b) => a.at - b.at)

  state.total = timeline.reduce((max, action) => Math.max(max, action.at + (action.duration ?? 0)), 0)
  state.startedAt = performance.now()
  state.progressFill.style.width = '0%'
  state.progressLabel.textContent = `0.0s / ${fmtTime(state.total)}`

  for (const action of timeline) {
    state.timers.push(setTimeout(() => ACTIONS[action.type]?.(action), action.at / state.speed))
  }

  state.raf = requestAnimationFrame(tickProgress)
}

const setSpeed = next => {
  state.speed = next

  for (const button of state.shell.querySelectorAll('[data-segment="speed"] button')) {
    button.classList.toggle('is-active', Number(button.dataset.value) === next)
  }
}

const setScale = next => {
  state.scale = next
  state.shell.style.setProperty('--scale', `${next}`)
  state.shell.style.setProperty('--stage-w', `${state.viewport.cols * state.viewport.cellWidth * next}px`)
  state.shell.style.setProperty('--stage-h', `${state.viewport.rows * state.viewport.lineHeight * next}px`)

  for (const button of state.shell.querySelectorAll('[data-segment="scale"] button')) {
    button.classList.toggle('is-active', Number(button.dataset.value) === next)
  }
}

const fitScale = () => {
  const margin = 96
  const baseW = state.viewport.cols * state.viewport.cellWidth
  const baseH = state.viewport.rows * state.viewport.lineHeight
  const maxW = Math.max(1, window.innerWidth - margin)
  const maxH = Math.max(1, window.innerHeight - 240)
  const fit = Math.max(1, Math.floor(Math.min(maxW / baseW, maxH / baseH)))

  return Math.max(1, Math.min(SCALES[SCALES.length - 1], fit))
}

const loadWorkflow = async name => {
  const url = new URL(window.location.href)
  url.searchParams.set('w', name)
  window.history.replaceState(null, '', url)

  try {
    const response = await fetch(`/api/workflow/${encodeURIComponent(name)}`)

    if (response.ok) {
      state.workflow = await response.json()
    }
  } catch {
    /* fall through */
  }

  await rebuild()
}

const buildOptions = () => {
  if (!catalog.length) {
    return ''
  }

  return catalog
    .map(({ name, title }) => {
      const selected = name === initial?.name ? ' selected' : ''

      return `<option value="${escapeHtml(name)}"${selected}>${escapeHtml(title)}</option>`
    })
    .join('')
}

const buildSegmented = (values, active) =>
  values
    .map(
      value =>
        `<button type="button" data-value="${value}" class="${value === active ? 'is-active' : ''}">${value}x</button>`
    )
    .join('')

const computeViewport = () => {
  const fromWorkflow = state.workflow.viewport ?? {}

  return {
    cellWidth: 8,
    cols: 80,
    lineHeight: 18,
    rows: 24,
    scale: 2,
    ...fromWorkflow
  }
}

const renderShell = () => {
  state.viewport = computeViewport()
  state.frameMode = (state.workflow.timeline ?? []).some(a => a.type === 'frame')
  state.frameTargets.clear()

  state.shell.style.setProperty('--cell-w', `${state.viewport.cellWidth}px`)
  state.shell.style.setProperty('--cols', `${state.viewport.cols}`)
  state.shell.style.setProperty('--line-h', `${state.viewport.lineHeight}px`)
  state.shell.style.setProperty('--rows', `${state.viewport.rows}`)
  state.shell.style.setProperty('--term-w', `${state.viewport.cols * state.viewport.cellWidth}px`)
  state.shell.style.setProperty('--term-h', `${state.viewport.rows * state.viewport.lineHeight}px`)

  state.shell.innerHTML = `
    <div class="showroom-stage">
      <div class="showroom-terminal">
        <div class="showroom-status" data-target="status">
          <span class="showroom-status-left"></span>
          <span class="showroom-status-right"></span>
        </div>
        <div class="showroom-body${state.frameMode ? ' is-frame-mode' : ''}"></div>
        <div class="showroom-composer" data-target="composer"></div>
      </div>
      <div class="showroom-overlays"></div>
    </div>
    <footer class="showroom-controls">
      <button type="button" data-action="restart" title="restart (R)">&#8635;</button>
      <span class="showroom-segmented" data-segment="scale">${buildSegmented(SCALES, state.scale)}</span>
      <span class="showroom-segmented" data-segment="speed">${buildSegmented(SPEEDS, state.speed)}</span>
      ${catalog.length > 1 ? `<select class="showroom-picker" data-action="picker">${buildOptions()}</select>` : ''}
      <span class="showroom-progress">
        <span data-role="time">0.0s / 0.0s</span>
        <div class="showroom-progress-track"><div class="showroom-progress-fill"></div></div>
      </span>
    </footer>
  `

  state.body = state.shell.querySelector('.showroom-body')
  state.composer = state.shell.querySelector('.showroom-composer')
  state.overlays = state.shell.querySelector('.showroom-overlays')
  state.statusLeft = state.shell.querySelector('.showroom-status-left')
  state.statusRight = state.shell.querySelector('.showroom-status-right')
  state.progressFill = state.shell.querySelector('.showroom-progress-fill')
  state.progressLabel = state.shell.querySelector('[data-role="time"]')

  state.shell.querySelector('[data-action="restart"]').addEventListener('click', play)

  for (const button of state.shell.querySelectorAll('[data-segment="speed"] button')) {
    button.addEventListener('click', () => setSpeed(Number(button.dataset.value)))
  }

  for (const button of state.shell.querySelectorAll('[data-segment="scale"] button')) {
    button.addEventListener('click', () => setScale(Number(button.dataset.value)))
  }

  const picker = state.shell.querySelector('[data-action="picker"]')

  if (picker) {
    picker.addEventListener('change', event => {
      void loadWorkflow(event.target.value)
    })
  }
}

const rebuild = async () => {
  renderShell()
  setScale(state.workflow.viewport?.scale ?? fitScale())
  play()
}

const mount = () => {
  state.shell = document.createElement('section')
  state.shell.className = 'showroom-shell'
  root.replaceChildren(state.shell)

  void rebuild().then(() => {
    requestAnimationFrame(() => state.shell.classList.add('is-mounted'))
  })

  window.addEventListener('keydown', event => {
    const key = event.key.toLowerCase()

    if (key === 'r') {
      play()
    } else if (key === '1' || key === '2' || key === '3') {
      setSpeed(SPEEDS[Number(key) - 1])
    }
  })
}

mount()

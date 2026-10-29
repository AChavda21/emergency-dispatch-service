// Audio alerts via the Web Audio API (no audio files to load).
// Browsers block sound until the user interacts with the page, so the engine
// is only created from a click on the audio toggle.

export class AudioAlerts {
    constructor() {
        this.ctx = null;
        this.enabled = false;
    }

    toggle() {
        if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
        if (this.ctx.state === "suspended") this.ctx.resume();
        this.enabled = !this.enabled;
        if (this.enabled) this.tone({ type: "sine", from: 660, to: 880, duration: 0.15 });
        return this.enabled;
    }

    play(incident) {
        if (!this.enabled || !this.ctx) return;
        if (incident.severity_flag) {
            // Two descending sawtooth sweeps: the original high-severity alarm, doubled
            this.tone({ type: "sawtooth", from: 880, to: 440, duration: 0.4 });
            this.tone({ type: "sawtooth", from: 880, to: 440, duration: 0.4, delay: 0.45 });
        } else if (incident.needs_human_review) {
            this.tone({ type: "triangle", from: 520, to: 520, duration: 0.18 });
            this.tone({ type: "triangle", from: 620, to: 620, duration: 0.18, delay: 0.22 });
        } else {
            this.tone({ type: "sine", from: 440, to: 440, duration: 0.4 });
        }
    }

    tone({ type, from, to, duration, delay = 0, volume = 0.05 }) {
        const start = this.ctx.currentTime + delay;
        const oscillator = this.ctx.createOscillator();
        const gain = this.ctx.createGain();

        oscillator.type = type;
        oscillator.frequency.setValueAtTime(from, start);
        if (to !== from) oscillator.frequency.exponentialRampToValueAtTime(to, start + duration);

        gain.gain.setValueAtTime(volume, start);
        gain.gain.exponentialRampToValueAtTime(0.001, start + duration);

        oscillator.connect(gain);
        gain.connect(this.ctx.destination);
        oscillator.start(start);
        oscillator.stop(start + duration);
    }
}

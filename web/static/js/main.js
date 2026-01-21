document.addEventListener('DOMContentLoaded', () => {
    const dateInput = document.getElementById('report-date');
    const refreshBtn = document.getElementById('refresh-btn');
    const currentTimeDisplay = document.getElementById('current-time');
    const loadBacktestBtn = document.getElementById('load-backtest-btn');

    // Initial load
    fetchData();
    fetchBacktest();

    refreshBtn.addEventListener('click', () => {
        fetchData(dateInput.value);
    });

    dateInput.addEventListener('change', () => {
        fetchData(dateInput.value);
    });

    loadBacktestBtn.addEventListener('click', () => {
        fetchBacktest();
    });

    function fetchData(date = null) {
        let url = '/api/data';
        if (date) {
            url += `?date=${date}`;
        }

        fetch(url)
            .then(response => response.json())
            .then(data => {
                updateDashboard(data);
            })
            .catch(error => console.error('Error fetching data:', error));
    }

    function fetchBacktest() {
        fetch('/api/backtest?days=30')
            .then(response => response.json())
            .then(data => {
                updateBacktest(data);
            })
            .catch(error => console.error('Error fetching backtest:', error));
    }

    function updateDashboard(data) {
        // Update Date Display
        currentTimeDisplay.textContent = `DATE: ${data.date}`;
        dateInput.value = data.date;

        // Update Top 5 Predictions
        const listEl = document.getElementById('predictions-list');
        listEl.innerHTML = '';

        if (data.predictions && data.predictions.length > 0) {
            data.predictions.slice(0, 5).forEach((pred, index) => {
                const item = document.createElement('div');
                item.className = 'prediction-item' + (index === 0 ? ' top-one' : '');
                item.innerHTML = `
                    <div class="pred-rank">#${index + 1}</div>
                    <div class="pred-info">
                        <h3>${pred.name}</h3>
                        <p>${pred.reason}</p>
                    </div>
                    <div class="pred-score">${pred.score.toFixed(4)}</div>
                `;
                listEl.appendChild(item);
            });
        } else {
            listEl.innerHTML = '<div class="no-data">今日暂无预测数据</div>';
        }

        // Update time
        const now = new Date();
        document.getElementById('update-time').textContent = 
            `更新于 ${now.getHours().toString().padStart(2,'0')}:${now.getMinutes().toString().padStart(2,'0')}`;
    }

    function updateBacktest(data) {
        if (data.status !== 'success') {
            document.getElementById('perf-win-rate').textContent = '暂无数据';
            document.getElementById('perf-beat-market').textContent = '--';
            document.getElementById('perf-total-return').textContent = '--';
            document.getElementById('perf-sharpe').textContent = '--';
            document.getElementById('history-tbody').innerHTML = 
                '<tr><td colspan="5" style="text-align:center;color:#888;">暂无回测数据</td></tr>';
            return;
        }

        const perf = data.performance;
        
        // Update performance cards (新的板块指标)
        document.getElementById('perf-win-rate').textContent = `${perf.win_rate || perf.overall_hit_rate || '--'}%`;
        document.getElementById('perf-beat-market').textContent = `${perf.beat_market_rate || '--'}%`;
        document.getElementById('perf-total-return').textContent = `${perf.total_return}%`;
        document.getElementById('perf-sharpe').textContent = perf.sharpe_ratio || '--';

        // Update hit rate bars (涨幅命中率)
        const hit1 = perf.hit_1pct || perf.top1_hit_rate || 0;
        const hit2 = perf.hit_2pct || perf.top3_hit_rate || 0;
        const hit3 = perf.hit_3pct || perf.top5_hit_rate || 0;

        document.getElementById('bar-hit1').style.width = `${hit1}%`;
        document.getElementById('rate-hit1').textContent = `${hit1}%`;
        document.getElementById('bar-hit2').style.width = `${hit2}%`;
        document.getElementById('rate-hit2').textContent = `${hit2}%`;
        document.getElementById('bar-hit3').style.width = `${hit3}%`;
        document.getElementById('rate-hit3').textContent = `${hit3}%`;

        // Update history table
        const tbody = document.getElementById('history-tbody');
        tbody.innerHTML = '';

        if (data.history && data.history.length > 0) {
            data.history.forEach(row => {
                const tr = document.createElement('tr');
                const hitIcon = row.is_hit === true ? '✅' : (row.is_hit === false ? '❌' : '⏳');
                const changeClass = row.actual_change_pct > 0 ? 'positive' : (row.actual_change_pct < 0 ? 'negative' : '');
                
                tr.innerHTML = `
                    <td>${row.predict_date}</td>
                    <td><strong>${row.sector_name}</strong></td>
                    <td>#${row.predict_rank}</td>
                    <td class="${changeClass}">${row.actual_change_pct !== null ? row.actual_change_pct + '%' : '--'}</td>
                    <td>${hitIcon}</td>
                `;
                tbody.appendChild(tr);
            });
        } else {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#888;">暂无历史记录</td></tr>';
        }
    }
});

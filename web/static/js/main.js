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
            document.getElementById('perf-hit-rate').textContent = '暂无数据';
            document.getElementById('perf-avg-return').textContent = '--';
            document.getElementById('perf-total-return').textContent = '--';
            document.getElementById('perf-limit-up').textContent = '--';
            document.getElementById('history-tbody').innerHTML = 
                '<tr><td colspan="6" style="text-align:center;color:#888;">暂无回测数据</td></tr>';
            return;
        }

        const perf = data.performance;
        
        // Update performance cards
        document.getElementById('perf-hit-rate').textContent = `${perf.overall_hit_rate}%`;
        document.getElementById('perf-avg-return').textContent = `${perf.avg_daily_return}%`;
        document.getElementById('perf-total-return').textContent = `${perf.total_return}%`;
        document.getElementById('perf-limit-up').textContent = perf.total_limit_up;

        // Update rank hit rate bars
        const top1 = perf.top1_hit_rate || 0;
        const top3 = perf.top3_hit_rate || 0;
        const top5 = perf.top5_hit_rate || 0;

        document.getElementById('bar-top1').style.width = `${top1}%`;
        document.getElementById('rate-top1').textContent = `${top1}%`;
        document.getElementById('bar-top3').style.width = `${top3}%`;
        document.getElementById('rate-top3').textContent = `${top3}%`;
        document.getElementById('bar-top5').style.width = `${top5}%`;
        document.getElementById('rate-top5').textContent = `${top5}%`;

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
                    <td>${row.actual_limit_up}</td>
                    <td>${hitIcon}</td>
                `;
                tbody.appendChild(tr);
            });
        } else {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#888;">暂无历史记录</td></tr>';
        }
    }
});

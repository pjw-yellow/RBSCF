function [g_f,h_l] = run_training(model_xf,xcf,use_sz,params,yf,small_filter_sz,frame,h_l)

    g_f = single(zeros(size(model_xf)));
    h_f = g_f;
    l_f = g_f;
    alpha    = 1;
    betha = 10;
    mumax = 10000;
    i = 1;
    
    T = prod(use_sz);
    S_xx0 = sum(conj(model_xf) .* model_xf, 3);
    S_xx = S_xx0 + sqrt(params.yta) * sum(conj(xcf) .* xcf, 3);
    u_f = model_xf + sqrt(params.yta) * xcf;
    S_uu = sum(conj(u_f) .* model_xf, 3);
    
    %   ADMM
    while (i <= params.admm_iterations)
        %   solve for G- please refer to the paper for more details
        B = S_xx + (T * alpha);
        S_lx = sum(conj(u_f) .* l_f, 3);
        S_hx = sum(conj(u_f) .* h_f, 3);
        g_f = (((1/(T*alpha)) * bsxfun(@times, yf, model_xf)) - ((1/alpha) * l_f) + h_f) - ...
            bsxfun(@rdivide,(((1/(T*alpha)) * bsxfun(@times, u_f, (S_uu .* yf))) - ((1/alpha) * bsxfun(@times, u_f, S_lx)) + (bsxfun(@times, u_f, S_hx))), B);
%         %%% ²»ÓÃsM
%         g_f = (model_xf .* conj(yf) + mu * T * h_f - T * l_f) ./(S_xx0 + params.yta * sum(conj(xcf) .* xcf, 3) + mu * T);
        
        %   solve for H
        if(frame==1)
        h = (T * ifft2((alpha*g_f) + l_f)) / ((alpha*T) + params.admm_lambda);   
        h_l=h;
        else
        h = (T * ifft2((alpha*g_f) + l_f) + params.mu * h_l) / ((alpha*T) + params.admm_lambda + params.mu);
        end
        if(i==params.admm_iterations)
             h_l = (1-0.025)*h_l+0.025*h;
        end
        [sx,sy,h] = get_subwindow_no_window(h, floor(use_sz/2) , small_filter_sz);
        t = single(zeros(use_sz(1), use_sz(2), size(h,3)));
        t(sx,sy,:) = h;
        h_f = fft2(t);
        %   update L
        l_f = l_f + (alpha * (g_f - h_f));
        
        %   update alpah- betha = 10.
        alpha = min(betha * alpha, mumax);
        i = i+1;
    end
    
end

